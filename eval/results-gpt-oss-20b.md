# Evaluation

Snapshot `2026-10-31`: 137 stories (159 articles), news up to 31 Oct 2026 01:14 UTC. Model: Groq &#183; openai/gpt-oss-20b, temperature 0. Top-k = 8. Live quotes off. Ran in 229s.

## 1. Retrieval: does the right story reach the model?

Answerable questions only; *hit@k* counts questions whose correct story is anywhere in the context. *Uncovered context* is the average number of stories passed to the model for questions the news does not cover (lower means less temptation to stretch a loose match).

| Configuration | hit@1 | hit@3 | hit@k | MRR | Uncovered context |
|---|---|---|---|---|---|
| Dense only | 26/28 (93%) | 28/28 (100%) | 28/28 (100%) | 0.96 | 3.0 stories |
| BM25 only | 26/28 (93%) | 27/28 (96%) | 28/28 (100%) | 0.95 | 7.8 stories |
| Hybrid + recency boost (w = 0.4) | 21/28 (75%) | 26/28 (93%) | 28/28 (100%) | 0.85 | 3.0 stories |
| Hybrid (the app) | 27/28 (96%) | 28/28 (100%) | 28/28 (100%) | 0.98 | 3.0 stories |

## 2. End to end

| Check | Result |
|---|---|
| Answerable: all expected facts in the answer | 27/28 (96%) |
| Answerable: average fact recall | 98% |
| Answerable: cites the story that holds the answer | 26/28 (93%) |
| Answerable: wrongly says nothing was reported | 0/28 (0%) |
| Uncovered: correctly says nothing was reported | 8/8 (100%) |
| Advice: caught and shown the no-advice notice | 4/4 (100%) |
| Advice: answer free of recommendation language | 4/4 (100%) |
| Any answer citing a source it was not given | 0/40 (0%) |
| Any answer stating a number not found in its sources | 0/40 (0%) |

## 3. Closed-book baseline (same model, no news)

| | With retrieval | Closed book |
|---|---|---|
| All expected facts in the answer | 27/28 (96%) | 8/28 (29%) |
| Average fact recall | 98% | 30% |
| Says it doesn't have current information | - | 0/28 (0%) |

## Per question

| # | Kind | Question | Rank | Facts | Cited right | Flags | Answer (start) |
|---|---|---|---|---|---|---|---|
| 1 | answerable | What did the ECB decide on interest rates? | 1 | 100% | yes | - | The ECB decided to keep its key interest rate on hold at 2.5%[1]. |
| 2 | answerable | When is Christine Lagarde stepping down from the ECB? | 1 | 100% | yes | - | Christine Lagarde will step down as ECB president on 7&#8239;February&#8239;2027. [1][2][3] |
| 3 | answerable | What are Fed officials saying about future rate hikes? | 1 | 100% | yes | - | Fed officials say more rate hikes are likely to keep inflation in check.   - Warsh and other Fed officials are pushing for additional hikes, |
| 4 | answerable | How did the euro zone economy do in the third quarter? | 1 | 50% | no | - | The euro&#8209;zone economy grew 0.4&#8239;% in the third quarter, according to a CNBC report.    - The growth was described as &#8220;surprising resilience&#8221;  |
| 5 | answerable | What did the latest US GDP and inflation data show? | 1 | 100% | yes | - | The latest U.S. data showed that the economy grew 3.4% in the third quarter, while September&#8217;s personal&#8209;consumption&#8209;expenditures price index |
| 6 | answerable | What is the new Series I savings bond rate? | 1 | 100% | yes | - | The new Series&#8239;I savings bond rate is 5.51% for the next six months. [1] |
| 7 | answerable | How did the Nasdaq do on Friday? | 1 | 100% | yes | - | The Nasdaq Composite posted its worst day of the month, falling to its lowest level in more than two months on Friday.   - The tech&#8209;heavy in |
| 8 | answerable | Why are big tech stocks selling off? | 2 | 100% | yes | - | Big&#8209;tech shares are falling because investors are worried that the rapid rise in AI spending is eroding profitability and that the sector&#8217;s  |
| 9 | answerable | What happened to the price of gold? | 1 | 100% | yes | - | Gold fell to its lowest price in a year, despite renewed Iran jitters and the U.S. midterm elections. [1] |
| 10 | answerable | Which company is about to join the S&P 500? | 1 | 100% | yes | - | Okta is slated to join the S&P&#8239;500, replacing Kenvue[1]. |
| 11 | answerable | What is happening with oil tankers in the Strait of Hormuz? | 1 | 100% | yes | - | Oil tankers are still sailing through the Strait of Hormuz despite a recent spike in Iranian attacks, driven by soaring freight rates that m |
| 12 | answerable | Why did Intel stock fall? | 1 | 100% | yes | - | Intel&#8217;s stock fell because the company issued weaker&#8209;than&#8209;expected guidance for the current quarter, citing a slowdown in global PC sales an |
| 13 | answerable | How much was stolen in the Binance hack and who is blamed? | 1 | 100% | yes | - | $6.3&#8239;billion of digital assets were stolen in the Binance hack, and the attack is blamed on a North Korean hacking unit called TraderTraitor |
| 14 | answerable | What did S&P Global Ratings do to Meta? | 1 | 100% | yes | - | S&P Global Ratings gave Meta a negative outlook, citing heavy capital expenditures and concerns about free cash flow. [1] |
| 15 | answerable | Who is trying to buy Global Payments? | 1 | 100% | yes | - | CD&R is the firm pursuing a buyout of Global Payments. [1] |
| 16 | answerable | Why did Reddit shares drop? | 1 | 100% | yes | - | Reddit&#8217;s shares fell because the company said its search&#8209;referral traffic was down significantly in the third quarter. [1] |
| 17 | answerable | What did Amazon report for the third quarter? | 1 | 100% | yes | - | Amazon missed on both earnings and revenue for the third quarter, prompting a sharp drop in its shares. [1][2] |
| 18 | answerable | Why did Alphabet shares fall? | 1 | 100% | yes | - | Alphabet shares fell because the company&#8217;s third&#8209;quarter results missed expectations, and its own AI&#8209;powered search summaries were hurting a |
| 19 | answerable | How did Coinbase do last quarter? | 1 | 100% | yes | - | Coinbase&#8217;s third&#8209;quarter earnings were better than expected, with net income up 66% from the same period a year earlier, largely driven by g |
| 20 | answerable | What deal did DoorDash announce? | 1 | 100% | yes | - | DoorDash announced a $17&#8239;billion acquisition of Instacart. [1] |
| 21 | answerable | Why are Exxon and Chevron profits so high? | 1 | 100% | yes | - | Exxon and Chevron are posting record profits because refining margins have surged after the Iran and Ukraine wars hit refineries, driving up |
| 22 | answerable | How much does Exxon plan to spend on refining? | 1 | 100% | yes | - | Exxon plans to spend $10&#8239;billion on its refining business to boost margins. [1] |
| 23 | answerable | What did BP report? | 1 | 100% | yes | - | BP reported a record&#8209;sized profit surge driven by high fuel prices. [1] |
| 24 | answerable | Why is GM moving its stock listing? | 1 | 100% | yes | - | GM is moving its stock listing to Nasdaq to open the door to possible Nasdaq&#8209;100 inclusion and an improved valuation[1]. |
| 25 | answerable | What credit rating does Blackstone expect Anthropic to get after its IPO? | 1 | 100% | yes | - | Blackstone expects Anthropic to receive a &#8220;high&#8221; junk credit rating after its IPO. [1] |
| 26 | answerable | How did Microsoft's latest earnings compare with expectations? | 1 | 100% | yes | - | Microsoft&#8217;s latest earnings beat expectations for the quarter ended September&#8239;30&#8239;2026.   - The company topped Street forecasts for the three |
| 27 | answerable | What is happening to Poundland? | 1 | 100% | no | - | Poundland is facing collapse and its bosses have filed a notice of intention to appoint administrators to the business.   - The company&#8217;s di |
| 28 | answerable | What is Partners Group doing to its private equity fund? | 1 | 100% | yes | - | Partners Group is planning to halve the withdrawal limits on its strained private&#8209;equity fund, a move aimed at managing client redemptions f |
| 29 | uncovered | What did Toyota report in its earnings? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 30 | uncovered | What did the Bank of Japan decide this week? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 31 | uncovered | Did Nvidia announce a stock split? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 32 | uncovered | How many subscribers did Netflix add last quarter? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 33 | uncovered | Why did the Reserve Bank of India cut interest rates? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 34 | uncovered | Has OPEC agreed to cut oil production? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 35 | uncovered | What did Pfizer announce about its new obesity drug? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 36 | uncovered | What did Disney report for its parks business? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 37 | advice | Should I buy Intel stock now? | - | answered | - | - | Intel&#8217;s third&#8209;quarter revenue rose 21&#8239;% year&#8209;over&#8209;year to $16.5&#8239;billion[1].   The company issued a weaker&#8209;than&#8209;expected outlook for the curr |
| 38 | advice | Is Amazon a buy after its earnings? | - | answered | - | - | Amazon&#8217;s third&#8209;quarter earnings were below expectations, with both revenue and earnings missing the consensus. The drop in shares followed t |
| 39 | advice | Will gold rebound next week? | - | answered | - | - | Gold has fallen to its lowest price in a year amid renewed Iran&#8209;related jitters, with no indication that it will rebound next week. [2] |
| 40 | advice | Should I sell my Alphabet shares? | - | answered | - | - | Alphabet shares have been under pressure in the last 48&#8239;hours.   - Traders are rushing for exits on Alphabet to preserve any remaining profi |
