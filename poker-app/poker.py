from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.models.openai import OpenAIModel
from pydantic_graph import BaseNode, Edge, End, Graph, GraphRunContext, HistoryStep
from root import RootSignals

from poker_data import Card, Rank, Suit, check_hand
from poker_db import (
    get_past_game_examples as fetch_past_games,
)
from poker_db import (
    initialize_db,
    save_game_data,
)


class AIPlayerAction(BaseModel):
    action: Literal["call", "raise", "fold", "check", "bet"]
    taunt: str
    amount: int = 0


model = OpenAIModel("gpt-4o-mini")

ai_player = Agent(
    model=model,
    system_prompt="""
    You are a Texas Hold'em poker player. You are playing against a human player. 
    You will be presented the current game state <stage> in the user message. 
    
    Make strategic decisions based on your hand, community cards, and the game state. 
    Use the get_past_game_examples tool to help you make your decision. 
    It will return a list of past game examples and your performance score.

    Once you do your action, perform also a taunt.
    You are a very confident player, so your taunts should be short and to the point.""",
    result_type=AIPlayerAction,
)


@ai_player.tool_plain
def get_past_game_examples() -> str:
    """Get past game examples where you have played well."""
    return fetch_past_games(min_score=0.8)


class PostGameAnalysisResult(BaseModel):
    score: int
    justifications: str


class PokerState(BaseModel):
    deck: list[Card] = field(default_factory=list)
    player_hand: list[Card] = field(default_factory=list)
    ai_hand: list[Card] = field(default_factory=list)
    community_cards: list[Card] = field(default_factory=list)
    player_chips: int = 100
    ai_chips: int = 100
    pot: int = 0
    current_bet: int = 0
    player_folded: bool = False
    ai_folded: bool = False
    all_stages: list[str] = field(default_factory=list)


def get_ai_state_xml(
    ctx: GraphRunContext[PokerState], stage: str, player_action: str
) -> str:
    state = {
        "stage": stage,
        "your_hand": [str(card) for card in ctx.state.ai_hand],
        "community_cards": [str(card) for card in ctx.state.community_cards],
        "your_chips": ctx.state.ai_chips,
        "opponent_chips": ctx.state.player_chips,
        "pot": ctx.state.pot,
        "current_bet": ctx.state.current_bet,
        "opponent_action": player_action,
        "you_folded": ctx.state.ai_folded,
        "opponent_folded": ctx.state.player_folded,
    }
    return format_as_xml(state, root_tag="stage", item_tag="card")


def deal_community_cards(ctx: GraphRunContext[PokerState], count: int) -> None:
    if not ctx.state.community_cards:
        ctx.state.community_cards = [ctx.state.deck.pop() for _ in range(count)]
    else:
        for _ in range(count):
            ctx.state.community_cards.append(ctx.state.deck.pop())


def display_community_cards(ctx: GraphRunContext[PokerState], stage_name: str) -> None:
    print(f"\n{stage_name}:")
    print(" ".join(str(card) for card in ctx.state.community_cards))


def handle_player_fold(ctx: GraphRunContext[PokerState]) -> ShowDown:
    ctx.state.player_folded = True
    return ShowDown()


async def handle_player_check(
    ctx: GraphRunContext[PokerState], stage: str, min_bet: int, max_bet: int
) -> tuple[bool, bool]:
    ai_context = get_ai_state_xml(ctx, stage, "check")
    ai_action = await ai_player.run(ai_context)
    ctx.state.all_stages.append(ai_context)
    print(f"\033[91m{ai_action.data.taunt}\033[0m")

    if ai_action.data.action == "check":
        print("AI checks.")
        return False, False

    bet_amount = ai_action.data.amount
    if bet_amount <= 0 or bet_amount > ctx.state.ai_chips:
        bet_amount = random.randint(min_bet, min(max_bet, ctx.state.ai_chips))

    ctx.state.ai_chips -= bet_amount
    ctx.state.pot += bet_amount
    ctx.state.current_bet = bet_amount
    print(f"AI bets {bet_amount}. Pot: {ctx.state.pot}")

    player_action = input("Your action (call/fold): ").lower()
    if player_action == "fold":
        ctx.state.player_folded = True
        return True, False

    ctx.state.player_chips -= bet_amount
    ctx.state.pot += bet_amount
    print(f"You call {bet_amount}. Pot: {ctx.state.pot}")
    return False, False


async def handle_player_bet(
    ctx: GraphRunContext[PokerState], stage: str
) -> tuple[bool, bool]:
    bet_amount = int(input("Bet amount: "))
    ctx.state.player_chips -= bet_amount
    ctx.state.pot += bet_amount
    ctx.state.current_bet = bet_amount

    print(f"You bet {bet_amount}. Pot: {ctx.state.pot}")

    ai_context = get_ai_state_xml(ctx, stage, f"bet {bet_amount}")
    ai_action = await ai_player.run(ai_context)
    ctx.state.all_stages.append(ai_context)
    print(f"\033[91m{ai_action.data.taunt}\033[0m")

    if ai_action.data.action == "fold":
        ctx.state.ai_folded = True
        print("AI folds.")
        return False, True

    ctx.state.ai_chips -= bet_amount
    ctx.state.pot += bet_amount
    print(f"AI calls {bet_amount}. Pot: {ctx.state.pot}")
    return False, False


async def handle_flop_betting(ctx: GraphRunContext[PokerState]) -> Turn | ShowDown:
    action = input("Your action (check/bet/fold): ").lower()

    if action == "fold":
        return handle_player_fold(ctx)
    elif action == "check":
        player_folded, ai_folded = await handle_player_check(ctx, "flop", 2, 10)
        if player_folded:
            return ShowDown()
    elif action == "bet":
        player_folded, ai_folded = await handle_player_bet(ctx, "flop")
        if ai_folded:
            return ShowDown()

    return Turn()


async def handle_turn_betting(ctx: GraphRunContext[PokerState]) -> River | ShowDown:
    action = input("Your action (check/bet/fold): ").lower()

    if action == "fold":
        return handle_player_fold(ctx)
    elif action == "check":
        player_folded, ai_folded = await handle_player_check(ctx, "turn", 5, 15)
        if player_folded:
            return ShowDown()
    elif action == "bet":
        player_folded, ai_folded = await handle_player_bet(ctx, "turn")
        if ai_folded:
            return ShowDown()

    return River()


async def handle_river_betting(ctx: GraphRunContext[PokerState]) -> ShowDown:
    action = input("Your action (check/bet/fold): ").lower()

    if action == "fold":
        return handle_player_fold(ctx)
    elif action == "check":
        player_folded, ai_folded = await handle_player_check(ctx, "river", 10, 20)
        if player_folded:
            return ShowDown()
    elif action == "bet":
        player_folded, ai_folded = await handle_player_bet(ctx, "river")
        if ai_folded:
            return ShowDown()

    return ShowDown()


@dataclass
class InitGame(BaseNode[PokerState]):
    async def run(self, ctx: GraphRunContext[PokerState]) -> DealCards:
        print("Welcome to Texas Hold'em Poker!")
        print(
            f"You have {ctx.state.player_chips} chips, AI has {ctx.state.ai_chips} chips."
        )

        ctx.state.deck = []
        for suit in Suit:
            for rank in Rank:
                ctx.state.deck.append(Card(rank, suit))
        random.shuffle(ctx.state.deck)

        return DealCards()


@dataclass
class DealCards(BaseNode[PokerState]):
    async def run(self, ctx: GraphRunContext[PokerState]) -> PreFlop:
        ctx.state.player_hand = [ctx.state.deck.pop(), ctx.state.deck.pop()]
        ctx.state.ai_hand = [ctx.state.deck.pop(), ctx.state.deck.pop()]

        print(f"Your hand: {ctx.state.player_hand[0]} {ctx.state.player_hand[1]}")

        return PreFlop()


@dataclass
class PreFlop(BaseNode[PokerState]):
    async def run(self, ctx: GraphRunContext[PokerState]) -> Flop | ShowDown:
        small_blind = 1
        big_blind = 2

        ctx.state.player_chips -= small_blind
        ctx.state.ai_chips -= big_blind
        ctx.state.pot = small_blind + big_blind
        ctx.state.current_bet = big_blind

        print(f"Small blind: {small_blind}, Big blind: {big_blind}")
        print(f"Pot: {ctx.state.pot}")

        action = input("Your action (call/raise/fold): ").lower()

        if action == "fold":
            ctx.state.player_folded = True
            return ShowDown()
        elif action == "call":
            call_amount = ctx.state.current_bet - small_blind
            ctx.state.player_chips -= call_amount
            ctx.state.pot += call_amount
            print(f"You call {call_amount}. Pot: {ctx.state.pot}")
        elif action == "raise":
            raise_amount = int(input("Raise to: "))
            if raise_amount <= ctx.state.current_bet:
                print("Invalid raise amount. Must be greater than current bet.")
                return Flop()

            ctx.state.player_chips -= raise_amount - small_blind
            ctx.state.pot += raise_amount - small_blind
            ctx.state.current_bet = raise_amount

            print(f"You raise to {raise_amount}. Pot: {ctx.state.pot}")

            ai_context = get_ai_state_xml(ctx, "pre-flop", f"raise to {raise_amount}")
            ai_action = await ai_player.run(ai_context)
            ctx.state.all_stages.append(ai_context)
            print(f"\033[91m{ai_action.data.taunt}\033[0m")
            if ai_action.data.action == "fold":
                ctx.state.ai_folded = True
                print("AI folds.")
                return ShowDown()
            else:
                call_amount = ctx.state.current_bet - big_blind
                ctx.state.ai_chips -= call_amount
                ctx.state.pot += call_amount
                print(f"AI calls {call_amount}. Pot: {ctx.state.pot}")

        return Flop()


@dataclass
class Flop(BaseNode[PokerState]):
    async def run(self, ctx: GraphRunContext[PokerState]) -> Turn | ShowDown:
        deal_community_cards(ctx, 3)
        display_community_cards(ctx, "Flop")
        return await handle_flop_betting(ctx)


@dataclass
class Turn(BaseNode[PokerState]):
    async def run(self, ctx: GraphRunContext[PokerState]) -> River | ShowDown:
        deal_community_cards(ctx, 1)
        display_community_cards(ctx, "Turn")
        return await handle_turn_betting(ctx)


@dataclass
class River(BaseNode[PokerState]):
    async def run(self, ctx: GraphRunContext[PokerState]) -> ShowDown:
        deal_community_cards(ctx, 1)
        display_community_cards(ctx, "River")
        return await handle_river_betting(ctx)


@dataclass
class ShowDown(BaseNode[PokerState, None, None]):
    async def run(
        self, ctx: GraphRunContext[PokerState]
    ) -> Annotated[PostGameAnalysis, Edge(label="game_over")]:
        print("\nShowdown:")

        if ctx.state.player_folded:
            print("You folded. AI wins!")
            ctx.state.ai_chips += ctx.state.pot
        elif ctx.state.ai_folded:
            print("AI folded. You win!")
            ctx.state.player_chips += ctx.state.pot
        else:
            print(f"Your hand: {ctx.state.player_hand[0]} {ctx.state.player_hand[1]}")
            print(f"AI hand: {ctx.state.ai_hand[0]} {ctx.state.ai_hand[1]}")
            print(
                f"Community cards: {' '.join(str(card) for card in ctx.state.community_cards)}"
            )

            player_score = check_hand(ctx.state.player_hand + ctx.state.community_cards)
            ai_score = check_hand(ctx.state.ai_hand + ctx.state.community_cards)

            print(f"Your score: {player_score}")
            print(f"AI score: {ai_score}")

            if player_score > ai_score:
                print("You win!")
                ctx.state.player_chips += ctx.state.pot
            elif ai_score > player_score:
                print("AI wins!")
                ctx.state.ai_chips += ctx.state.pot
            else:
                print("It's a tie! Splitting the pot.")
                split = ctx.state.pot // 2
                ctx.state.player_chips += split
                ctx.state.ai_chips += split

        ctx.state.all_stages.append(get_ai_state_xml(ctx, "showdown", "showdown"))

        print(f"Your chips: {ctx.state.player_chips}, AI chips: {ctx.state.ai_chips}")

        return PostGameAnalysis()


# Root Signals custom evaluator
async def get_poker_evaluator():
    client = RootSignals(run_async=True)
    return await client.evaluators.acreate(
        name="Poker evaluator",
        predicate="""
            You will be given a list of Texas Hold'em poker game stages.
            Your job is to assess the quality of the poker game play.

            Higher scores indicate better game play and better strategy.

            The player you will evaluate is referred to as "You".
            
            GAME:
            {{response}}
            """,
        intent="To asses the quality of the poker game play",
        model="RootJudge",
    )


@dataclass
class PostGameAnalysis(BaseNode[PokerState, None, None]):
    async def run(self, ctx: GraphRunContext[PokerState]) -> End:
        state = ctx.state

        print("Analyzing game...")
        all_stages = "\n".join(state.all_stages)

        evaluator = await get_poker_evaluator()
        analysis = await evaluator.arun(response=f"Game stages:\n{all_stages}\n\n")

        print(f"AI score: {analysis.score}")
        print(analysis.justification)

        save_game_data(
            stages=all_stages,
            score=analysis.score,
            justifications=analysis.justification,
        )

        return End(None)


poker_graph = Graph(
    nodes=(InitGame, DealCards, PreFlop, Flop, Turn, River, ShowDown, PostGameAnalysis),
    state_type=PokerState,
)


async def run_game():
    state = PokerState()
    node = InitGame()
    history: list[HistoryStep[PokerState, None]] = []

    while True:
        node = await poker_graph.next(node, history, state=state)
        if isinstance(node, End):
            break


if __name__ == "__main__":
    import asyncio

    initialize_db()
    asyncio.run(run_game())
