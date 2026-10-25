# Answers to the questions we expect

These are the answers we prepared for questions after the talk. `make_script.py` appends them to `script.md` and fills in every number from `_variables.yml`. Each answer is a short list of full sentences, so that any of us can give it.

**What do you mean by a clock?**

- A clock is the quantity that sets when a network groks.
- The term comes from the title of Kim's paper on the weight-decay clock.
- We compare three: the shape of the kernel, its scale or the weight norm, and the weight-decay rate.

**Why 0.99 for memorisation but 0.95 for grokking?**

- We take the levels from Khanh and coauthors, who use 0.99 for memorisation and check their result at test accuracy 0.80, 0.90 and 0.95.
- We report 0.95 and repeated every analysis at 0.80 and 0.90.
- The signs of the scaling exponents a and b hold at every level.

**Why do you divide by the square root of the width?**

- The factors one over root N and one over root D are the NTK parameterisation of Jacot and coauthors.
- They give the network and its kernel a limit as the width grows, so the kernels of networks of different widths are on the same scale and can be compared.

**Why is the NTK parameterisation a limitation?**

- Jacot and coauthors note that these factors reduce the influence of each weight when the width is large, so wider networks stay closer to the lazy regime.
- The width effect we measure therefore belongs to this parameterisation.
- Another parameterisation scales the updates differently with width, so the width exponent a could change.

**Why do you use the logarithm of R?**

- We tried R and log R, among other non-linear forms, and log R predicted best of the forms of R.
- R also spans more than a decade across widths, so its logarithm puts the widths on one scale.
- TODO: confirm with Zain which forms were compared, and whether log S was tried.

**Is the kernel on 53 test pairs really the network's kernel?**

- We use the kernel of the summed logits, divided by the square root of p, on the 53 test pairs of each run.
- Mohamadi and coauthors show that this sum of logits kernel converges to the full kernel at initialisation, and it is far cheaper to compute.
- The kernel of all p logits may behave differently, which is one of our limitations.

**What does censored mean, and why keep those runs?**

- A run that has not grokked by step 200,000 is kept, and we record only that its grokking step is later than 200,000.
- Dropping those runs would keep only the fast runs and bias the fits.
- Survival models use censored runs through the likelihood, following Kalbfleisch and Prentice.
- In {{< var NeightNoGrok >}} of the 144 cells no seed groks within the budget.

**Alignment has p = {{< var EthreeNhundredAP >}}. Is it not significant?**

- Alignment is tied to grokking a little, since its p is below 0.05.
- Its SD ratio is only {{< var EthreeNhundredARatio >}}, close to one, so its spread hardly changes at the wrong step.
- Its median barely moves, from {{< var NsixAzero >}} at the start to {{< var NsixAgrok >}} at grokking, so we do not call it a threshold or a clock.

**Does the network grok once it reaches a certain shape change?**

- No, reaching it is not enough.
- In {{< var NeightAboveRMedianMemorised >}} of the {{< var NeightNoGrokMemorised >}} cells where a seed memorises but none groks, a seed still reaches the median shape change at grokking of its width.
- The shape change at grokking also falls with width, from {{< var NsevenRWFifty >}} at width 50 to {{< var NsevenRWSixteenhundred >}} at width 1600.

**What does gamma tell you?**

- Gamma lets width and weight decay interact on the log scale.
- Our estimate is {{< var EsixMainG >}}, with an interval from {{< var EsixMainGLo >}} to {{< var EsixMainGHi >}}.
- The interval includes zero, so we do not claim its sign.

**Do the exponents depend on which runs you fit?**

- Yes, so we fit the range where every seed groks, weight decay from ten to the minus five to ten to the minus four.
- Adding the next weight decay, three times ten to the minus four, moves b to {{< var EsixWideB >}}.
- So we quote a and b for that range only.

**Could a better model predict the grokking time from the kernel?**

- On an unseen alpha, boosted trees on all three statistics reach a concordance of {{< var EeightXgb >}}, and a recurrent network on their history reaches {{< var EeightGru >}}, where 0.5 is chance.
- Trees given the width and the weight decay reach {{< var EeightRef >}}, and the weight decay alone reaches {{< var EeightDecayref >}}.
- So the settings predict better than any kernel model, and we do not claim that the kernel beats the weight decay.
- The kernel is useful because it can be measured in any network, while the effective regularisation of a real network cannot.

**Do shape and scale interact?**

- We do not claim that they do.
- That analysis was cut from the report, so the talk does not use it.

**Why does MNIST give a different answer?**

- On MNIST the delay comes from scaling up the initial weights, and weight decay has to shrink them to a fixed norm before the network generalises, following Liu and coauthors.
- So the time is one over the weight decay, the norm at grokking is {{< var EnineteenNc >}} within {{< var EnineteenNcRel >}}, and width drops out.
- Our reading is that a fixed kernel cannot solve modular addition, as Mohamadi and coauthors prove, so there the kernel must change shape first and width enters.
- MNIST also changes the network and how the delay is made, so we cannot say which change gives the different answer.

**Why the squared error and not cross-entropy?**

- The baseline of Kumar and coauthors uses the squared error, and we reproduce it.
- Our proposal planned a cross-entropy arm, which we did not run.
- Under cross-entropy, Khanh and coauthors find that the weight norm acts through the scale of the logits, and we have not tested that setting.

**Why is the statistic called R if it measures shape?**

- R is one minus the cosine between the kernel now and at the start, which is the kernel distance of Fort and coauthors.
- Our earlier drafts called it rotation, and a rotation is one kind of change of shape.
- It is zero when the kernel has only been rescaled.

**What is new in this work?**

- We split the movement of the kernel into scale and shape and measure both.
- We add a placebo test that separates a real threshold from a statistic that barely moves.
- We cross laziness, width and weight decay in one grid, on two tasks.
- We predict the grokking time from quantities that any network exposes.
