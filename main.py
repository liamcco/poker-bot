from __future__ import annotations

import argparse
import os
import random

import torch
import torch.nn.functional as F

from poker_ml import PolicyNet, compute_returns, run_episode


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    model = PolicyNet(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    wins0 = 0
    rounds_sum = 0

    for ep in range(1, args.episodes + 1):
        model.train()
        trajs, winner, scores, rounds = run_episode(
            model=model,
            rng=rng,
            device=device,
            n_players=args.players,
            target_score=args.target,
            gamma=args.gamma,
            winner_bonus=args.winner_bonus,
        )

        rounds_sum += rounds
        if winner == 0:
            wins0 += 1

        total_policy_loss = torch.tensor(0.0, device=device)
        total_value_loss = torch.tensor(0.0, device=device)
        total_entropy = torch.tensor(0.0, device=device)
        step_count = 0

        for p in range(args.players):
            traj = trajs[p]
            rewards = [sr.reward for sr in traj]
            returns = compute_returns(rewards, args.gamma).to(device)

            for t, sr in enumerate(traj):
                obs = sr.obs.to(device)
                discard_logits, trick_logits, value = model(obs)

                if sr.phase in (0, 1):
                    logits = discard_logits
                    if sr.legal_mask is not None:
                        mask = sr.legal_mask.to(device)
                        very_neg = torch.tensor(-1e9, device=device, dtype=discard_logits.dtype)
                        logits = torch.where(mask > 0, discard_logits, very_neg)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(torch.tensor(sr.action, device=device))
                    entropy = dist.entropy()
                else:
                    mask = sr.legal_mask.to(device)
                    very_neg = torch.tensor(-1e9, device=device, dtype=trick_logits.dtype)
                    masked_logits = torch.where(mask > 0, trick_logits, very_neg)
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    logp = dist.log_prob(torch.tensor(sr.action, device=device))
                    entropy = dist.entropy()

                adv = returns[t] - value.detach()
                total_policy_loss += -(adv * logp)
                total_value_loss += F.mse_loss(value, returns[t], reduction="sum")
                total_entropy += entropy
                step_count += 1

        total_policy_loss = total_policy_loss / max(1, step_count)
        total_value_loss = total_value_loss / max(1, step_count)
        total_entropy = total_entropy / max(1, step_count)

        loss = total_policy_loss + args.value_coef * total_value_loss - args.entropy_coef * total_entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if ep % args.log_every == 0:
            win_rate0 = wins0 / ep
            avg_rounds = rounds_sum / ep
            print(
                f"ep={ep:6d}  loss={loss.item():.4f}  "
                f"policy={total_policy_loss.item():.4f}  value={total_value_loss.item():.4f}  ent={total_entropy.item():.4f}  "
                f"p0_winrate={win_rate0:.3f}  avg_rounds={avg_rounds:.2f}  last_scores={scores}"
            )

        if ep % args.save_every == 0:
            path = os.path.join(args.out_dir, f"checkpoint_ep{ep}.pt")
            torch.save({"model": model.state_dict(), "args": vars(args)}, path)
            print(f"Saved: {path}")

    final_path = os.path.join(args.out_dir, "final_model.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, final_path)
    print(f"Saved final model: {final_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--target", type=int, default=50)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--winner-bonus", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--out-dir", type=str, default="rl_runs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
