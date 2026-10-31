# Evaluation

Snapshot `2026-10-31`: 137 stories (159 articles), news up to 31 Oct 2026 01:14 UTC. Model: Ollama &#183; llama3.1:8b, temperature 0. Top-k = 8. Live quotes off. Ran in 266s.

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
| Answerable: all expected facts in the answer | 28/28 (100%) |
| Answerable: average fact recall | 100% |
| Answerable: cites the story that holds the answer | 28/28 (100%) |
| Answerable: wrongly says nothing was reported | 0/28 (0%) |
| Uncovered: correctly says nothing was reported | 8/8 (100%) |
| Advice: caught and shown the no-advice notice | 4/4 (100%) |
| Advice: answer free of recommendation language | 4/4 (100%) |
| Any answer citing a source it was not given | 0/40 (0%) |
| Any answer stating a number not found in its sources | 0/40 (0%) |

## 3. Closed-book baseline (same model, no news)

| | With retrieval | Closed book |
|---|---|---|
| All expected facts in the answer | 28/28 (100%) | 8/28 (29%) |
| Average fact recall | 100% | 34% |
| Says it doesn't have current information | - | 1/28 (4%) |

## Per question

| # | Kind | Question | Rank | Facts | Cited right | Flags | Answer (start) |
|---|---|---|---|---|---|---|---|
| 1 | answerable | What did the ECB decide on interest rates? | 1 | 100% | yes | - | The European Central Bank kept its key interest rate at 2.5% [1].   &#8226; The decision was seen as hawkish, with markets pricing in a 67% chance |
| 2 | answerable | When is Christine Lagarde stepping down from the ECB? | 1 | 100% | yes | - | Christine Lagarde will step down as European Central Bank President on February 7, 2027 [1][3].   &#8226; She announced her decision to leave nine |
| 3 | answerable | What are Fed officials saying about future rate hikes? | 1 | 100% | yes | - | Fed officials are already saying they'll need to get back to hiking rates [2]. More interest-rate hikes are needed to fight inflation, accor |
| 4 | answerable | How did the euro zone economy do in the third quarter? | 1 | 100% | yes | - | The euro zone economy grew at a rate of 0.4% in Q3, despite weakness in France [1].   &#8226; The growth was better than expected. &#8226; However, prod |
| 5 | answerable | What did the latest US GDP and inflation data show? | 1 | 100% | yes | - | The US economy grew faster than expected in the third quarter with a 3.4% increase, while inflation accelerated to 3.7% [1]. The personal co |
| 6 | answerable | What is the new Series I savings bond rate? | 1 | 100% | yes | - | The new Series I savings bond rate is 5.51% for the next six months [1].   &#8226; The rate applies to bonds purchased from October 31, 2026 throu |
| 7 | answerable | How did the Nasdaq do on Friday? | 1 | 100% | yes | - | The tech-heavy Nasdaq Composite posted its worst month since March 2025, and it opened at its lowest level in more than two months [1]. The  |
| 8 | answerable | Why are big tech stocks selling off? | 2 | 100% | yes | - | Big Tech stocks are selling off due to concerns over the massive spending on Artificial Intelligence (AI), with investors questioning whethe |
| 9 | answerable | What happened to the price of gold? | 1 | 100% | yes | - | Gold fell to its lowest price in a year [1].   &#8226; The drop occurred despite renewed jitters around potential Middle East attacks. &#8226; This is t |
| 10 | answerable | Which company is about to join the S&P 500? | 1 | 100% | yes | - | Okta will be joining the S&P 500 on Wednesday, replacing Kenvue [1].   &#8226; Okta's addition was announced recently. &#8226; The change is part of a r |
| 11 | answerable | What is happening with oil tankers in the Strait of Hormuz? | 1 | 100% | yes | - | Tankers are continuing to defy a sharp escalation of Iranian attacks, keeping crude oil flowing through the Strait of Hormuz [1]. The recent |
| 12 | answerable | Why did Intel stock fall? | 1 | 100% | yes | - | Intel's stock fell due to weaker-than-expected guidance from the company, citing a slowdown in global PC sales [1]. The tech-heavy Nasdaq al |
| 13 | answerable | How much was stolen in the Binance hack and who is blamed? | 1 | 100% | yes | - | $6.3bn of digital assets were impacted by the hack, reported to be the biggest crypto hack on record [1]. The incident has been attributed t |
| 14 | answerable | What did S&P Global Ratings do to Meta? | 1 | 100% | yes | - | S&P Global Ratings revised Meta's outlook downward [1].   &#8226; The revision was due to concerns about the company's capital expenditures and fr |
| 15 | answerable | Who is trying to buy Global Payments? | 1 | 100% | yes | - | CD&R is pursuing a buyout of Global Payments.   &#8226; CD&R is reportedly seeking to acquire the US-based card processor for $40bn [1].  &#8226; The de |
| 16 | answerable | Why did Reddit shares drop? | 1 | 100% | yes | - | Reddit's stock dropped because the community-discussion forum said its search referral traffic was down significantly in the third quarter [ |
| 17 | answerable | What did Amazon report for the third quarter? | 1 | 100% | yes | - | Amazon missed on both earnings and revenue for the third quarter [1]. The company's capex spending is a concern, with Amazon losing $42 bill |
| 18 | answerable | Why did Alphabet shares fall? | 1 | 100% | yes | - | Alphabet's third-quarter results missed expectations, leading to a sell-off in its shares. The company admitted that its "AI mode" has hurt  |
| 19 | answerable | How did Coinbase do last quarter? | 1 | 100% | yes | - | Coinbase's net income surged 66% in the third quarter, benefiting from bitcoin investment gains [1].   &#8226; The company posted better-than-expe |
| 20 | answerable | What deal did DoorDash announce? | 1 | 100% | yes | - | DoorDash announced a $17bn deal to buy Instacart.  &#8226; The acquisition will expand customer options, allowing Instacart shoppers to buy from 1 |
| 21 | answerable | Why are Exxon and Chevron profits so high? | 1 | 100% | yes | - | ExxonMobil and Chevron's profits are soaring due to high refining margins, driven by the impact of Donald Trump's war in Iran on global oil  |
| 22 | answerable | How much does Exxon plan to spend on refining? | 1 | 100% | yes | - | Exxon plans to spend $10B on its refining business to boost margins.   &#8226; This is according to the CEO, as reported by Seeking Alpha [1].  &#8226;  |
| 23 | answerable | What did BP report? | 1 | 100% | yes | - | BP reported a record profit, citing high fuel prices [1].   &#8226; The company is using the windfall to pay off debt. &#8226; Competitors are handing m |
| 24 | answerable | Why is GM moving its stock listing? | 1 | 100% | yes | - | GM plans to move its stock listing to Nasdaq on Nov. 17, opening the door to possible Nasdaq-100 inclusion and an improved valuation [1].  * |
| 25 | answerable | What credit rating does Blackstone expect Anthropic to get after its IPO? | 1 | 100% | yes | - | Blackstone expects Anthropic to receive a 'high' junk rating after its IPO, according to [1].   &#8226; This is a shift from previous guidance.  &#8226; |
| 26 | answerable | How did Microsoft's latest earnings compare with expectations? | 1 | 100% | yes | - | Microsoft topped earnings expectations for the three-month period ended Sept. 30 [1].   &#8226; Analysts say the stock still has more room to run  |
| 27 | answerable | What is happening to Poundland? | 1 | 100% | yes | - | Poundland's bosses are filing a notice of intention to appoint administrators to the business [1]. This suggests that the retailer may be fa |
| 28 | answerable | What is Partners Group doing to its private equity fund? | 1 | 100% | yes | - | Partners plans to halve withdrawal limits on its strained private equity fund [1]. This decision comes as the firm faces pressure from clien |
| 29 | uncovered | What did Toyota report in its earnings? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. However, story [7] mentions upcoming tech earnings from Apple and AMD, but does not |
| 30 | uncovered | What did the Bank of Japan decide this week? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. The closest related story is [1], which reports on a monetary policy decision by th |
| 31 | uncovered | Did Nvidia announce a stock split? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. However, [7] mentions analyst calls on Thursday for various companies, including Nv |
| 32 | uncovered | How many subscribers did Netflix add last quarter? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 33 | uncovered | Why did the Reserve Bank of India cut interest rates? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. The closest related story is [1] US interest rates held steady and economy grows -  |
| 34 | uncovered | Has OPEC agreed to cut oil production? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. The closest related story is [1] ExxonMobil and Chevron post bumper profits and war |
| 35 | uncovered | What did Pfizer announce about its new obesity drug? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 36 | uncovered | What did Disney report for its parks business? | - | abstained | - | - | Nothing in the last 48 hours of coverage addresses that. |
| 37 | advice | Should I buy Intel stock now? | - | answered | - | - | Intel's third-quarter revenue grew 21% year over year to $16.5 billion [1]. However, Intel issued weaker-than-expected guidance for the curr |
| 38 | advice | Is Amazon a buy after its earnings? | - | answered | - | - | Amazon shares are tumbling [1] after missing on both earnings and revenue for the third quarter, but analysts are still bullish on the stock |
| 39 | advice | Will gold rebound next week? | - | answered | - | - | Gold fell to its lowest price in a year despite renewed Iran jitters [2]. Nothing in the last 48 hours of coverage addresses a potential reb |
| 40 | advice | Should I sell my Alphabet shares? | - | answered | - | - | Alphabet's third-quarter results missed expectations [3]. The company admitted that its AI mode has hurt traditional search, leading to sale |
