import asyncio
import concurrent.futures
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import torch
import re
from copy import deepcopy
from openai import OpenAI
from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

PROMPT_GRM = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client question displayed below.

[Client Question]
{conv_his}

[The Start of Chatbot A's Response]
{response_A}
[The End of Chatbot A's Response]


[The Start of Chatbot B's Response]
{response_B}
[The End of Chatbot B's Response]

Output your final verdict by strictly following this format:

<critics>
[Provide a brief summary of your reasoning for the choice]
</critics>

<choice>
[[A]]
</choice>

Note: Use [[A]] if A is better, or [[B]] if B is better."""

MAX_RETRIES = 5

def parse_result_default(llm_output: str) -> dict:
    choice_patterns = [
        r"<choice>\s*\[\[\s*([AB])\s*\]\]\s*</choice>",
        r"<choice>.*?\[\[\s*([AB])\s*\]\].*?</choice>",
        r"<choice>\s*([AB])\s*</choice>",
        r"\[\[\s*([AB])\s*\]\]",
        r"\[\s*([AB])\s*\]",
    ]
    choice = None
    for pattern in choice_patterns:
        matches = re.findall(pattern, llm_output, re.DOTALL)
        if matches:
            choice = matches[-1].upper()
            break
    if choice not in ["A", "B"]:
        choice = None
    return {"choice": choice}


async def call_grm_async(
        context_str: str,
        response_a: str,
        response_b: str,
        grm_client: OpenAI,
        grm_name: str,
        executor: concurrent.futures.Executor,
) -> dict:
    input_text = PROMPT_GRM.format(
        conv_his=context_str,
        response_A=response_a,
        response_B=response_b,
    )
    messages = [{"role": "user", "content": input_text}]

    loop = asyncio.get_running_loop()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            def _make_request(msgs=messages, model=grm_name, client=grm_client):
                return client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    max_tokens=8192,
                )

            completion = await loop.run_in_executor(
                executor,
                _make_request,
            )
            llm_output = completion.choices[0].message.content

            if llm_output is None:
                print(
                    f"[GRM] Attempt {attempt}/{MAX_RETRIES}: got None response",
                    flush=True
                )
                continue

            result = parse_result_default(llm_output)

            if result.get("choice") is not None:
                return result

            print(
                f"[GRM] Attempt {attempt}/{MAX_RETRIES}: parse failure, "
                f"raw output snippet: {llm_output[:200]!r}",
                flush=True
            )

        except Exception as e:
            import traceback
            print(
                f"[GRM] Attempt {attempt}/{MAX_RETRIES} failed with "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                flush=True
            )

    print(f"[GRM] All {MAX_RETRIES} retries exhausted, defaulting to tie.", flush=True)
    return {"choice": None}


async def pairwise_compare_async(
        context_str: str,
        response_a: str,
        response_b: str,
        idx_a: int,
        idx_b: int,
        grm_client: OpenAI,
        grm_name: str,
        executor: concurrent.futures.Executor,
) -> Tuple[int, int, float, float]:
    """
    Compare two responses. Returns (winner_idx, loser_idx, winner_score, loser_score).
    """
    result = await call_grm_async(
        context_str, response_a, response_b, grm_client, grm_name, executor
    )
    choice = result.get("choice", None)

    if choice == "A":
        return idx_a, idx_b, 1.0, 0.0
    elif choice == "B":
        return idx_b, idx_a, 1.0, 0.0
    else:
        if random.random() < 0.5:
            return idx_a, idx_b, 0.5, 0.5
        else:
            return idx_b, idx_a, 0.5, 0.5


class DoubleEliminationTournament:
    """
    Double-Elimination Tournament for ranking responses within a group.
    """

    def __init__(
            self,
            participants: List[dict],
            context_str: str,
            grm_client: OpenAI,
            grm_name: str,
            executor: concurrent.futures.Executor,
    ):
        """
        ...
        """
        self.participants = participants
        self.context_str = context_str
        self.grm_client = grm_client
        self.grm_name = grm_name
        self.executor = executor
        self.n = len(participants)

        self.losses = {p["original_idx"]: 0 for p in participants}
        self.total_score = {p["original_idx"]: 0.0 for p in participants}
        self.num_matches = {p["original_idx"]: 0 for p in participants}
        self.elimination_round = {p["original_idx"]: -1 for p in participants}
        self.response_map = {p["original_idx"]: p["response_str"] for p in participants}

    async def run(self) -> Dict[int, float]:
        if self.n <= 1:
            return {self.participants[0]["original_idx"]: 1.0} if self.n == 1 else {}
        indices = [p["original_idx"] for p in self.participants]
        random.shuffle(indices)
        winners_bracket, losers_bracket, elimination_order = list(indices), [], []
        current_round = 0
        while len(winners_bracket) + len(losers_bracket) > 1:
            current_round += 1
            wb_next, wb_losers = await self._run_bracket_round(winners_bracket, current_round)
            winners_bracket = wb_next
            for loser_idx in wb_losers:
                self.losses[loser_idx] = 1
                losers_bracket.append(loser_idx)
            if len(losers_bracket) > 1:
                lb_next, lb_eliminated = await self._run_bracket_round(losers_bracket, current_round)
                losers_bracket = lb_next
                for elim_idx in lb_eliminated:
                    self.losses[elim_idx] = 2
                    self.elimination_round[elim_idx] = current_round
                    elimination_order.append(elim_idx)
            if len(winners_bracket) == 1 and len(losers_bracket) == 1:
                wb_champ, lb_champ = winners_bracket[0], losers_bracket[0]
                winner, loser, w_score, l_score = await self._compare(wb_champ, lb_champ, current_round + 1)
                self._update_scores(wb_champ, lb_champ, w_score, l_score, winner)
                if winner == wb_champ:
                    self.elimination_round[lb_champ] = current_round + 1
                    elimination_order.append(lb_champ)
                    self.elimination_round[wb_champ] = current_round + 2
                    elimination_order.append(wb_champ)
                else:
                    self.losses[wb_champ] = 1
                    current_round += 1
                    winner2, loser2, w_score2, l_score2 = await self._compare(wb_champ, lb_champ, current_round + 1)
                    self._update_scores(wb_champ, lb_champ, w_score2, l_score2, winner2)
                    if winner2 == lb_champ:
                        self.elimination_round[wb_champ] = current_round + 1
                        elimination_order.append(wb_champ)
                        self.elimination_round[lb_champ] = current_round + 2
                        elimination_order.append(lb_champ)
                    else:
                        self.elimination_round[lb_champ] = current_round + 1
                        elimination_order.append(lb_champ)
                        self.elimination_round[wb_champ] = current_round + 2
                        elimination_order.append(wb_champ)
                winners_bracket, losers_bracket = [], []
                break
            if len(winners_bracket) == 0 and len(losers_bracket) > 1:
                while len(losers_bracket) > 1:
                    current_round += 1
                    lb_next, lb_eliminated = await self._run_bracket_round(losers_bracket, current_round)
                    losers_bracket = lb_next
                    for elim_idx in lb_eliminated:
                        self.losses[elim_idx] = 2
                        self.elimination_round[elim_idx] = current_round
                        elimination_order.append(elim_idx)
                if losers_bracket:
                    self.elimination_round[losers_bracket[0]] = current_round + 1
                    elimination_order.append(losers_bracket[0])
                break
        all_indices, already_ordered = set(indices), set(elimination_order)
        remaining = all_indices - already_ordered
        for r in remaining:
            elimination_order.append(r)
        n, rank_scores = len(elimination_order), {}
        for rank_position, idx in enumerate(elimination_order):
            rank_scores[idx] = rank_position / (n - 1) if n > 1 else 1.0
        rank_scores = self._apply_tiebreakers(elimination_order, rank_scores)
        return rank_scores

    async def _run_bracket_round(self, bracket: List[int], round_num: int) -> Tuple[List[int], List[int]]:
        winners, losers, match_tasks = [], [], []
        bye_participant = bracket[-1] if len(bracket) % 2 == 1 else None
        bracket_to_pair = bracket[:-1] if bye_participant is not None else bracket
        for i in range(0, len(bracket_to_pair), 2):
            idx_a, idx_b = bracket_to_pair[i], bracket_to_pair[i + 1]
            match_tasks.append(self._compare(idx_a, idx_b, round_num))
        match_results = await asyncio.gather(*match_tasks)
        for winner_idx, loser_idx, w_score, l_score in match_results:
            pair_indices = None
            for j in range(0, len(bracket_to_pair), 2):
                if bracket_to_pair[j] in (winner_idx, loser_idx) and bracket_to_pair[j + 1] in (winner_idx, loser_idx):
                    pair_indices = (bracket_to_pair[j], bracket_to_pair[j + 1])
                    break
            idx_a, idx_b = pair_indices if pair_indices else (winner_idx, loser_idx)
            self._update_scores(idx_a, idx_b, w_score, l_score, winner_idx)
            winners.append(winner_idx)
            losers.append(loser_idx)
        if bye_participant is not None:
            winners.append(bye_participant)
        return winners, losers

    async def _compare(
            self, idx_a: int, idx_b: int, round_num: int
    ) -> Tuple[int, int, float, float]:
        """Compare two participants asynchronously."""
        response_a = self.response_map[idx_a]
        response_b = self.response_map[idx_b]

        return await pairwise_compare_async(
            self.context_str,
            response_a,
            response_b,
            idx_a,
            idx_b,
            self.grm_client,
            self.grm_name,
            self.executor,
        )

    def _update_scores(self, idx_a: int, idx_b: int, w_score: float, l_score: float, winner_idx: int):
        if winner_idx == idx_a:
            self.total_score[idx_a] += w_score
            self.total_score[idx_b] += l_score
        else:
            self.total_score[idx_b] += w_score
            self.total_score[idx_a] += l_score
        self.num_matches[idx_a] += 1
        self.num_matches[idx_b] += 1

    def _apply_tiebreakers(self, elimination_order: List[int], rank_scores: Dict[int, float]) -> Dict[int, float]:
        round_groups = defaultdict(list)
        for idx in elimination_order:
            round_groups[self.elimination_round[idx]].append(idx)
        for er, group in round_groups.items():
            if len(group) <= 1: continue

            def avg_score(idx):
                return self.total_score[idx] / self.num_matches[idx] if self.num_matches[idx] > 0 else 0.0

            group_sorted = sorted(group, key=avg_score)
            positions = sorted([elimination_order.index(idx) for idx in group])
            n = len(elimination_order)
            for new_pos_idx, pos in enumerate(positions):
                idx = group_sorted[new_pos_idx]
                rank_scores[idx] = pos / (n - 1) if n > 1 else 1.0
        return rank_scores


def get_score(data, tokenizer, grm_client=None, grm_name=None):
    """
    Compute pairwise GRM scores for all data items.
    """
    groups = defaultdict(list)
    for i in range(len(data)):
        data_item = data[i]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        eos_token = tokenizer.eos_token
        if eos_token and response_str.endswith(eos_token):
            response_str = response_str[: -len(eos_token)]
        extra_info = data_item.non_tensor_batch.get("extra_info", None)
        context_str = ""
        if extra_info is not None and extra_info.get("context", None) is not None:
            context = extra_info.get("context", None)
            if isinstance(context, list):
                for message in context:
                    context_str += f"{message.get('role', 'unknown')}:\n{message.get('content', '')}\n"
            elif isinstance(context, str):
                context_str = context
        else:
            raise ValueError("extra_info is None or extra_info['context'] is None!")
        prompt_key = tuple(valid_prompt_ids.tolist())
        groups[prompt_key].append(
            {"original_idx": i, "response_str": response_str, "context_str": context_str}
        )

    results = [None] * len(data)

    # Define your desired max concurrency here
    MAX_CONCURRENCY = 200

    async def process_all_groups(executor: concurrent.futures.Executor):
        group_tasks = []
        for prompt_key, group_items in groups.items():
            group_tasks.append(process_single_group(group_items, executor))
        pbar = tqdm(total=len(group_tasks), desc="[GRM] Pairwise scoring groups", dynamic_ncols=True)
        async def _wrap(coro):
            result = await coro
            pbar.update(1)
            return result
        wrapped = [_wrap(task) for task in group_tasks]
        await asyncio.gather(*wrapped)
        pbar.close()

    async def process_single_group(group_items: List[dict], executor: concurrent.futures.Executor):
        n = len(group_items)
        if n == 1:
            results[group_items[0]["original_idx"]] = 0.5
            return

        context_str = group_items[0]["context_str"]

        if n == 2:
            idx_a, idx_b = group_items[0]["original_idx"], group_items[1]["original_idx"]
            response_a, response_b = group_items[0]["response_str"], group_items[1]["response_str"]
            winner_idx, loser_idx, w_score, l_score = await pairwise_compare_async(
                context_str, response_a, response_b, idx_a, idx_b,
                grm_client, grm_name, executor,  # Pass executor
            )
            results[winner_idx] = w_score
            results[loser_idx] = l_score
            return

        participants = [{"original_idx": item["original_idx"], "response_str": item["response_str"]} for item in
                        group_items]

        tournament = DoubleEliminationTournament(
            participants=participants,
            context_str=context_str,
            grm_client=grm_client,
            grm_name=grm_name,
            executor=executor,  # Pass executor
        )
        rank_scores = await tournament.run()
        for idx, score in rank_scores.items():
            results[idx] = score

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            loop.run_until_complete(process_all_groups(executor))
        except RuntimeError:
            asyncio.run(process_all_groups(executor))

    for i in range(len(data)):
        if results[i] is None:
            results[i] = 0.0

    return results


@register("pairwise_grm")
class PairwiseGRMRewardManager:
    """The reward manager."""

    def __init__(
            self,
            tokenizer,
            num_examine,
            compute_score=None,
            reward_fn_key="data_source",
            **reward_kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.grm_client = OpenAI(
            base_url=reward_kwargs["grm_host"],
            api_key="any-key",
            timeout=60 * 10,
        )
        self.grm_name = reward_kwargs["grm_name"]

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        results = get_score(
            data,
            tokenizer=self.tokenizer,
            grm_client=self.grm_client,
            grm_name=self.grm_name,
        )
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            result = results[i]
            score: float
            if isinstance(result, dict):
                score = result["score"]
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
            reward = score
            reward_tensor[i, valid_response_length - 1] = reward
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

