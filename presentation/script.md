# Script for the G2 seminar

This script is written by `make_script.py` from the speaker notes in `slides.qmd`, on 26 October 2026. Do not edit it by hand. Change the notes in the deck and run the script again.

Each slide gives its number in the browser, its budget and the running total. "Say" is what to speak, at about 150 words a minute. "Point at" says how to read the figure. "If asked" holds detail for questions, which is not part of the timed talk. The answers to the questions we expect are at the end.

| Turn | Slides | Time |
|---|---|---|
| 1 | 1 to 4 | 1:20 |
| 2 | 5 to 9 | 2:15 |
| 3 | 10 to 14 | 1:50 |
| 4 | 15 to 18 | 2:00 |
| 5 | 19 to 26 | 3:15 |

## Turn 1 (1:20)

### Slide 1. Title: Scale or Shape? (0:15, total 0:15)

*Message:* we study what sets the delay in grokking by watching the tangent kernel.

**Say**

- We are group G2, and our project is about grokking, where a network generalises long after it fits its training data.
- We tracked the network's tangent kernel during training to find out what sets the length of that delay.

### Act 2. Grokking (0:05, total 0:20)

**Say**

- What sets this delay is disputed, and we test the published answers.

### Slide 3. A network can fit its training data long before it generalises (0:35, total 0:55)

*Message:* grokking is a long gap between fitting the training data and generalising.

**Say**

- This is one of our baseline networks learning modular addition.
- Training accuracy, in grey, reaches 0.99 early, and we call that step memorisation.
- Test accuracy, in black, stays near zero and climbs much later, and the step where it reaches 0.95 is the grokking step T.
- Over five seeds, the median memorisation step is 2,983 and the median grokking step is 23,000.
- That gap is grokking, and the rest of the talk asks what sets its length.

**Point at**

- The hollow marker is memorisation and the filled marker is the grokking step.

**If asked**

- We take the levels from Khanh and coauthors, who use 0.99 for memorisation and check their result at test accuracy 0.80, 0.90 and 0.95.
- We report 0.95 and repeated every analysis at 0.80 and 0.90, and the signs of our scaling exponents hold at every level.

### Slide 4. The task is modular addition, a standard test bed for grokking (0:25, total 1:20)

*Message:* the task is small and standard, so we can measure the kernel exactly.

**Say**

- Our task is modular addition with modulus 23.
- The network sees two numbers and must output their sum mod 23, so 20 plus 5 gives 2.
- Each run trains on 476 of the 529 pairs and must fill in the other 53, the outlined cells.
- This task is a standard test bed in the theory of grokking, and our setup is the baseline of Kumar and coauthors.

**If asked**

- We divide by the square root of N because this is the NTK parameterisation of Jacot and coauthors.
- Those factors give the network and its kernel a limit as the width grows, so networks of different widths have kernels on the same scale.
- The network has a ReLU and no biases, and its input is the one-hot vectors of a and b side by side, so D is 46.
- It is trained on the squared error by full-batch gradient descent with learning rate 100.

## Turn 2 (2:15)

### Act 5. The tangent kernel (0:05, total 1:25)

**Say**

- To answer that we need one tool, the tangent kernel.

### Slide 6. The tangent kernel shows which inputs the network treats alike (0:35, total 2:00)

*Message:* the kernel is the network's own similarity between inputs, and it shows which features the network learns with.

**Say**

- Suppose we take one gradient step to fix the output on one input, x.
- The weights move, so the output on every other input moves too.
- For another input x-prime, that move is set by the inner product of the two gradients, which is the tangent kernel.
- Near its start the network is a linear model on these gradient features, and if the kernel stays fixed, training ends at kernel regression with it.
- So the kernel is a snapshot of the features the network learns with.

**If asked**

- Section 5 of Jacot and coauthors shows that in the infinite-width limit the trained network equals kernel ridge regression with the tangent kernel as the ridge goes to zero.
- Their result has a second term that comes from the network at the start, and our centred network starts at zero, so that term vanishes.
- The linear model is defined in Section 1.1 of Chizat, Oyallon and Bach.

### Slide 7. Grokking may be the move from a fixed kernel to one that learns features (0:30, total 2:30)

*Message:* alpha sets how long a network stays lazy, and the lazy to rich account says grokking happens when the kernel starts to move.

**Say**

- In the lazy regime the kernel barely moves, so the network acts like kernel regression with the kernel it started with.
- In the rich regime the kernel changes, so the network learns new features.
- Kumar and coauthors propose that grokking is the move from lazy to rich.
- Their knob is alpha, which scales the output, and a larger alpha lets the weights move less, so the network stays lazy for longer.
- In our runs, the lazier the network, the later it groks.

**Point at**

- The plot shows test accuracy at three values of alpha, and at alpha 2 no seed groks within 200,000 steps.

**If asked**

- The centred predictor and the learning rate divided by alpha squared are from Kumar and coauthors, Appendix 8.1.
- Chizat, Oyallon and Bach show in Section 1.2 that scaling the output by a large factor makes training lazy.

### Slide 8. A kernel can grow or change shape, and a distance cannot tell which (0:35, total 3:05)

*Message:* a large kernel distance can come from growth alone, so we split it.

**Say**

- The usual evidence that the kernel changed is its distance from where it started.
- Think of the kernel as an ellipse whose axes are the directions the network cares about.
- It can grow and point the same way, or it can turn, and both give the same distance.
- The squared distance splits exactly into a scale part and a shape part, and in our baseline 90% of it at grokking is scale.
- So a large distance alone does not show that the network learned new features.

**If asked**

- The split comes from expanding the squared norm of the difference, and the backup slides give the steps.

### Slide 9. Three numbers say how the kernel moved (0:30, total 3:35)

*Message:* one number for size, one for shape and one for agreement with the labels, and only the first sees the size.

**Say**

- Scale, in blue, is the size of the kernel relative to the start.
- Shape, in orange, flattens each kernel into one long vector and takes one minus the cosine between now and the start.
- Alignment, in green, asks whether the kernel treats inputs as alike when they share an answer.
- Shape and alignment ignore the size, so a kernel that only grows moves scale alone.
- These colours stay the same for the rest of the talk.

**If asked**

- We measure the kernel of the summed logits divided by the square root of p, on the 53 test pairs of each run.
- Mohamadi and coauthors show that this sum of logits kernel converges to the full kernel at initialisation, and it is far cheaper to compute.
- The cosine is exact for matrices: the Frobenius inner product is the dot product of the flattened matrices, so Cauchy-Schwarz holds.
- Both kernels are positive semidefinite, so their inner product is not negative and shape lies between 0 and 1.
- Shape is zero only when the kernel is a positive multiple of the start, and any other change raises it, including a change in the proportions of the eigenvalues.
- The shape statistic is the kernel distance of Fort and coauthors.
- The alignment is the centred alignment of Cortes, Mohri and Rostamizadeh, who show that a kernel with high alignment to the labels admits a good predictor.

## Turn 3 (1:50)

### Act 10. What sets the clock? (0:05, total 3:40)

**Say**

- Now we can ask which kind of kernel movement sets the grokking time.

### Slide 11. Every published clock is scale-like (0:25, total 4:05)

*Message:* the published clocks are all scale-like, and none measures shape.

**Say**

- By a clock we mean the quantity that sets when a network groks.
- One account says the network groks once weight decay has shrunk the weight norm enough.
- Another says doubling the weight decay halves the time.
- The third rests on the kernel distance, which mostly records growth.
- So every account is scale-like, and none measures the shape of the kernel.

**If asked**

- Under cross-entropy, Khanh and coauthors find that the weight norm acts through the scale of the logits.

### Slide 12. In theory, weight decay rescales and width slows the change of shape (0:30, total 4:35)

*Message:* the theory gives the two clocks different predictions for width and weight decay, which we test with the exponents a and b.

**Say**

- Lewkowycz and Gur-Ari wrote down how the kernel changes under weight decay.
- Weight decay appears only in the first term, which changes the scale and never the shape.
- Only the second term, T, changes the shape, and it starts out of order one over the width.
- So if scale sets the clock, b is minus one.
- If shape sets the clock, wider networks grok later and weight decay matters little, which is our own prediction.

**Point at**

- On the left, weight decay shrinks the kernel and keeps its axes.
- On the right, the shape term turns it, and at width 1600 the turn is a sixteenth of the turn at width 100.

**If asked**

- Here k is the degree to which the output scales when every weight is scaled, and our network has k equal to 2 before the centring.
- The centred network is not homogeneous, so the equation is a guide for us and not a theorem.
- Lewkowycz and Gur-Ari credit the order one over N of T at the start to Dyer and Gur-Ari.

### Slide 13. Our hypothesis: the shape of the kernel sets the grokking time (0:30, total 5:05)

*Message:* our hypothesis is that shape sets the clock, and each test can tell the accounts apart.

**Say**

- Our hypothesis is that the shape of the kernel sets the grokking time.
- Each test separates the accounts in a different way.
- Tier 1 asks whether one statistic takes the same value at every grokking step, and we expect shape.
- Tier 2 asks how the time scales, and we expect wider networks to grok later and weight decay to matter little.
- Tier 3 asks which statistic, read early, predicts the time best, and we expect shape.

**If asked**

- In Tier 3 we test only quantities measured on the network.
- We set the weight decay ourselves, so we know it exactly, but a general network does not report its effective regularisation, while its tangent kernel can be measured.
- The cells marked "our prediction" give a prediction the source does not state, which we derived from it.
- For example, the weight-decay papers give one over the weight decay but say nothing about width, so "width does not matter" is ours.

### Slide 14. We trained 720 networks across laziness, width and weight decay (0:20, total 5:25)

*Message:* one grid crosses all three knobs, and slow runs are kept as censored data.

**Say**

- We trained 720 networks across alpha, width and weight decay, with five seeds per cell.
- Runs that never grok within 200,000 steps are kept as censored times, where we only know the step is later.
- At the baseline we reproduce the grokking steps of Kumar and coauthors.

**If asked**

- Dropping the runs that never grok would keep only the fast runs, so survival models use them through the likelihood instead.
- The fixed settings are p of 23, 90 percent of the pairs for training, full-batch gradient descent with learning rate 100 and the squared error.
- Each seed has its own split, and the runs took 18.9 hours of compute in all.

## Turn 4 (2:00)

### Act 15. Results (0:05, total 5:30)

**Say**

- Which clock does the grokking time follow?

### Slide 16. Tier 1: at a fixed width, shape is the statistic most tied to grokking (0:45, total 6:15)

*Message:* at a fixed width, the shape statistic marks the grokking step most closely.

**Say**

- Okay, so first we test whether grokking happens when a statistic hits a fixed value.
- Think of water, which boils at 100 degrees in any pot.
- If grokking had a boiling point, the statistic would read the same in every run at the moment that run groks.
- But a thermometer stuck on one number also reads the same every time.
- So we also read each run at the step where some other run grokked.
- If the statistic is a real boiling point, those readings should be much more spread out.
- Shape spreads out the most, 1.71 times.
- Scale spreads 1.29 times, and alignment hardly at all.

**Point at**

- In each panel the left box is the true grokking step and the right box is the placebo.

**If asked**

- The p values are <0.0005 for shape, <0.0005 for scale and 0.029 for alignment.
- Alignment is tied to grokking a little, but its ratio is close to one, so we do not call it a threshold.
- A standard deviation carries the units of its statistic, so we compare statistics only through the ratio and p.
- The placebo SD is the median over 2,000 shuffles in which no run keeps its own step.

### Slide 17. No statistic is a fixed threshold, so shape is not the whole clock (0:25, total 6:40)

*Message:* the tie between shape and grokking holds only within one width, so shape is not a universal threshold.

**Say**

- But water only boils at 100 degrees at sea level.
- Up a mountain it boils at a lower temperature.
- Width acts like the mountain.
- Once we change the width, no statistic has the same value at grokking.
- Wide networks grok with much less change of shape, and yet they grok later.
- So shape alone does not decide when a network groks.

**If asked**

- At grokking, R falls from 0.108 at width 50 to 0.019 at width 1600.
- In 8 of the 34 cells where a seed memorises but none groks, a seed still reaches the median shape change at grokking of its width, so reaching that shape is not enough to grok.
- Alignment barely moves: its median is 0.398 at the start and 0.406 at grokking.

### Slide 18. Tier 2: the grokking time follows both width and weight decay (0:45, total 7:25)

*Message:* both width and weight decay set the time, so neither single clock holds.

**Say**

- Next, how does the time scale with width and weight decay?
- Each clock predicts a power law, a straight line on log axes, so we fit log T as a plane in log width and log weight decay.
- The shape clock predicts a positive a and b near zero, and the weight-decay clock predicts a near zero and b of minus one.
- We find a = +0.24 and b = -0.63, and both intervals exclude zero.
- So the fit takes the sign of a from the shape clock and the sign of b from the weight-decay clock, which is a mixed clock.
- Gamma's interval includes zero, so we do not claim an interaction.

**If asked**

- The backup slide shows the grokking time at each width, with the slope each clock predicts.
- Sigma, 0.34, is the spread of log T between runs that the plane does not explain.
- The fit uses the range where every seed groks, and adding the next weight decay, three times ten to the minus four, moves b to -0.21, so the exponents depend on the range.
- Both logarithms are centred at the middle of the grid, so a and b are the slopes there.
- Wider networks memorise sooner, so the width effect is not a delay in fitting the training data.

## Turn 5 (3:15)

### Slide 19. Tier 3: shape is the best single predictor, and scale adds what shape lacks (0:40, total 8:05)

*Message:* shape carries the most information about when a network will grok, scale adds to it, and alignment adds little.

**Say**

- Can the statistics still forecast when a run will grok?
- It is like guessing a runner's marathon time from their first few kilometres.
- We read them at step 2,430, before any run groks, and predict settings the model has never seen.
- Alignment explains almost nothing.
- Shape alone explains about a third, and adding scale takes it to a half.

**If asked**

- We tried R as it is and log R, and log R predicted better in every model on held-out cells, and the backup slide shows why.
- At step 2,430, 46% of the runs that grok have R below 0.01, while the largest R is 0.13.
- On its own scale those runs bunch near zero, so a model can hardly tell them apart, while on a log scale they spread out and the grokking time falls close to a straight line.
- A straight line in R alone explains 0.29 of log T, and a straight line in log R explains 0.37.
- Kernel ridge on R alone can bend to follow the curve, so it reaches 0.34 against 0.35 with log R, and the gap is small.
- The gap is large once the statistics are combined: all three reach 0.47 with R as it is and 0.55 with log R.
- With R as it is, the interaction of scale and shape also does not show, so which pairs of statistics seem to interact depends on the scale of R.
- On an unseen alpha the gain is smaller: a straight line in the three statistics ranks runs a little better with R as it is, and log R helps once scale can bend as well.
- The settings predict better than any kernel model: on an unseen alpha, trees on the width and weight decay reach a concordance of 0.79, and weight decay alone 0.72.
- On that test, boosted trees on all three statistics reach 0.70 and a recurrent network on their history 0.76, so we do not claim that the kernel beats the weight decay.
- The kernel ridge regression uses a Gaussian kernel, and whole cells are held out because the seeds of a cell are close copies.
- We do not claim that shape and scale interact, because that analysis was cut from the report.

### Slide 20. Tier 3: a curve beats a straight line, and kernel ridge fits best (0:30, total 8:35)

*Message:* a curve predicts much better than a straight line, and the most flexible models add little on top.

**Say**

- A straight line through the three statistics explains 0.36.
- Letting it bend for scale, with a squared term, jumps it to 0.50.
- Kernel ridge predicts a run from the runs most like it, like pricing a house from similar houses.
- It does best, at 0.55.

**Point at**

- The dashed line is boosted trees given width, alpha and weight decay, which a real network does not report.

**If asked**

- The numbers come from model_fitting/4_nonlinear_exploration.ipynb, Parts 3 and 7, on the same 355 runs as the last slide, and its kernel ridge matches the report.
- Boosted trees do well on other inputs: 0.74 on the 25 early readings of the statistics and 0.93 on the settings. The notebook reads the history as mostly a record of the settings.
- On an unseen alpha, trees on the three statistics rank runs at a concordance of 0.70, better than any statistic alone.
- Each gain above 0.50 is about 0.02, which is close to our noise level, and seed noise sets a ceiling of 0.96.
- Without shape, kernel ridge falls from 0.55 to 0.23, so shape carries the most information.
- Kernel ridge fits $\log T$ minus its mean, with the sum over the training runs $i$, and its penalty and $\gamma$ are chosen by cross-validation over whole cells.
- The trees are XGBoost on squared error, with depth 1 to 4 chosen on held-out training cells.

### Slide 21. A curve changes the slope, and a product links two statistics (0:30, total 9:05)

*Message:* a curve lets the slope of a statistic change along the way, a product lets one statistic set the slope of another, and the best simple formula is a straight line plus a square of scale.

**Say**

- A straight line says each step up in a statistic always moves the time by the same amount.
- A curve lets that amount change, like a road that is steep and then flattens.
- A product means how much scale matters depends on shape.
- In our data that adds very little.

**If asked**

- The spline has 5 knots and degree 3, which gives 7 basis functions per statistic, and the weights $c_j$ are fitted by ridge regression.
- The square is the simplest curve, since its slope $\beta + 2\beta' z$ changes in a straight line.
- Shape enters on a log scale, so the straight line in $z_R$ is already a curve in $R_t$.
- The betas are fitted by least squares on the training cells of each fold, and the notebook did not save them, so we quote only the form and its score.

### Slide 22. Tier 4: on MNIST, weight decay alone sets the time (0:35, total 9:40)

*Message:* on a second task, weight decay alone sets the time through the weight norm, and shape plays no part.

**Say**

- To see whether this carries over, we repeated Tiers 1 and 2 on MNIST, in the setup of Omnigrok.
- Here the initial weights are scaled up, and weight decay must shrink them before the network generalises.
- The time is one over the weight decay, width drops out, and at grokking the weight norm is the same within 3%.
- Our reading is that MNIST never forces the kernel to change shape, while on modular addition a fixed kernel cannot solve the task.

**If asked**

- The fit gives b = -1.00 and a = -0.02, with a residual of only 0.03.
- The weight norm at grokking is 2.11 times the default initial norm over 314 runs, which is the law of Liu and coauthors, Section 2.
- Width still sets the shape change, since R at grokking falls from 0.045 to 0.002 between widths 200 and 800, while the time does not move.
- Width 100 is left out of the fits because its test accuracy at memorisation is already 0.72, which we read as a limit of capacity at 1,000 images.
- The grokking level is 0.80 because no run reaches 0.95 within the budget, and without weight decay no run reaches the level.
- The learning rate is reset from the curvature every 10,000 steps, which sets the memorisation step.
- We did not repeat Tier 3, because with a residual of 0.03 the time is a function of the settings alone.
- The placebo test cannot separate the statistics here, because each is a function of the step and the grokking times span more than a decade.

### Act 23. What we found (0:05, total 9:45)

**Say**

- So what sets the grokking time?

### Slide 24. Limitations (0:20, total 10:05)

*Message:* the claims hold for this task and parameterisation, on the runs that grok.

**Say**

- We used one task and the NTK parameterisation, where wider networks stay lazier, so the width effect may belong to this setup.
- In 70 of the 144 cells no seed groks, and the exponents depend on the cells fitted.
- The settings predict better than the kernel, and MNIST changes three things at once.

**If asked**

- Jacot and coauthors note that the NTK factors reduce the influence of each weight when the width is large, which is why wider networks stay lazier here.
- Another parameterisation could change the width exponent, so we do not claim a value of a beyond this setup.
- Each cell has 5 seeds, and each event is known only to the nearest saved step.
- MNIST shows which clock sets the time once a delay exists, and not whether shape can cause one.
- The alignment uses the kernel of the summed logits, and the kernel of all p logits may behave differently.

### Slide 25. Grokking runs on a mixed clock, and shape is its best single reading (0:30, total 10:35)

*Message:* grokking runs on a mixed clock, in which shape is the most informative statistic and scale also matters.

**Say**

- First, the usual kernel distance mostly records growth, so split it before reading it as feature learning.
- Second, on modular addition shape is the most informative statistic, but it is not enough alone and it is not a fixed threshold.
- Third, no account holds alone, ours included, because width, through shape, and weight decay, through scale, both set the time.
- On MNIST, where shape is not needed, weight decay alone sets it.

**If asked**

- What is new is the split into scale and shape, the placebo test, the crossed grid on two tasks, and prediction from quantities that any network exposes.
- A next step is to test whether the early rate of shape change falls as one over the width.

### Slide 26. Thank you (0:05, total 10:40)

**Say**

- Thank you, and we are happy to take questions.

## Answers to the questions we expect

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

- We fitted R as it is and log R in straight lines and in kernel ridge, and log R predicted better on held-out cells.
- R also spans more than a decade across widths, so its logarithm puts the widths on one scale.
- Kernel ridge on all three statistics reaches 0.55 with log R and 0.47 with R as it is.
- We did not try log S. S entered on its own scale, and a square term and splines let its curve bend.

**Is the kernel on 53 test pairs really the network's kernel?**

- We use the kernel of the summed logits, divided by the square root of p, on the 53 test pairs of each run.
- Mohamadi and coauthors show that this sum of logits kernel converges to the full kernel at initialisation, and it is far cheaper to compute.
- The kernel of all p logits may behave differently, which is one of our limitations.

**What does censored mean, and why keep those runs?**

- A run that has not grokked by step 200,000 is kept, and we record only that its grokking step is later than 200,000.
- Dropping those runs would keep only the fast runs and bias the fits.
- Survival models use censored runs through the likelihood, following Kalbfleisch and Prentice.
- In 70 of the 144 cells no seed groks within the budget.

**Alignment has p = 0.029. Is it not significant?**

- Alignment is tied to grokking a little, since its p is below 0.05.
- Its SD ratio is only 1.09, close to one, so its spread hardly changes at the wrong step.
- Its median barely moves, from 0.398 at the start to 0.406 at grokking, so we do not call it a threshold or a clock.

**Does the network grok once it reaches a certain shape change?**

- No, reaching it is not enough.
- In 8 of the 34 cells where a seed memorises but none groks, a seed still reaches the median shape change at grokking of its width.
- The shape change at grokking also falls with width, from 0.108 at width 50 to 0.019 at width 1600.

**What does gamma tell you?**

- Gamma lets width and weight decay interact on the log scale.
- Our estimate is -0.16, with an interval from -0.37 to +0.05.
- The interval includes zero, so we do not claim its sign.

**Do the exponents depend on which runs you fit?**

- Yes, so we fit the range where every seed groks, weight decay from ten to the minus five to ten to the minus four.
- Adding the next weight decay, three times ten to the minus four, moves b to -0.21.
- So we quote a and b for that range only.

**Could a better model predict the grokking time from the kernel?**

- On an unseen alpha, boosted trees on all three statistics reach a concordance of 0.70, and a recurrent network on their history reaches 0.76, where 0.5 is chance.
- Trees given the width and the weight decay reach 0.79, and the weight decay alone reaches 0.72.
- So the settings predict better than any kernel model, and we do not claim that the kernel beats the weight decay.
- The kernel is useful because it can be measured in any network, while the effective regularisation of a real network cannot.

**Do shape and scale interact?**

- We do not claim that they do.
- That analysis was cut from the report, so the talk does not use it.

**Why does MNIST give a different answer?**

- On MNIST the delay comes from scaling up the initial weights, and weight decay has to shrink them to a fixed norm before the network generalises, following Liu and coauthors.
- So the time is one over the weight decay, the norm at grokking is 2.11 within 3%, and width drops out.
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
