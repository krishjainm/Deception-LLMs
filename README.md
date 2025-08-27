# Deception Circuits in Large Language Models

## Overview
This project investigates whether **deception circuits** can be discovered and causally manipulated inside the reasoning traces of large language models (LLMs).  

When a model is incentivized to produce deceptive outputs, what differences emerge in its internal representations and multi-step reasoning compared to truthful outputs?  
Do circuit-level interventions learned in one deception context generalize to others (for example: games, roleplaying, sandbagging, reward-hacking)?  

We combine interpretability tools like **linear probes, sparse autoencoders, activation patching, and steering** to explore deception as a *causal* phenomenon in neural activations.

---

## Motivation
- **Problem:** LLMs can engage in deceptive reasoning even after RLHF alignment, yet current methods mostly detect dishonesty at the output level.  
- **Why It Matters:** If deception circuits remain hidden, AI may become untrustworthy in high-stakes domains like law, medicine, or finance.  
- **Our Approach:** Incentivize deception through structured tasks (poker, Mafia, sandbagging, roleplay) and analyze internal activations to uncover, test, and intervene in deception-related circuits.  
