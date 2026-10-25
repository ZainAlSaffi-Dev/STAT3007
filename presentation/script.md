# Script for the G2 seminar

This is the spoken script of the talk. It is built from the speaker notes in `slides.qmd`, with every number filled in from `_variables.yml` as it stood on 25 October 2026. The deck remains the source. If a note or a number changes there, change it here as well.

The talk has 21 slides and five act slides, for 10:00 in all. The turns follow the possible split for five speakers in `PLAN.md`, section 5. The group has not yet decided who takes which turn. At about 150 words a minute, each turn's text fits its time.

Each slide gives its number, its budget and the running total. "Say" is the text to speak. "If asked" holds extra detail for questions, which is not part of the timed talk.

## Turn 1. The title and grokking (1:40)

### Slide 1. Title: Scale or Shape? (0:15, total 0:15)

Say: We are group G2, and our project is about grokking. Grokking is when a network generalises long after it fits the training data. We tracked the neural tangent kernel during training with three measurements. We wanted to see what predicts the delay between memorising the training data and generalising.

### Act 1. Grokking (0:05, total 0:20)

Say: A network fits its training data early. Why does it take so much longer to work on data it has not seen?

### Slide 2. A network can fit its training data long before it generalises (0:35, total 0:55)

Say: This graph shows one of our baseline networks. Over five seeds, training accuracy reaches 0.99 by a median step of 2,983. Test accuracy only reaches 0.95 at a median step of 23,000. The delay between these two events is grokking. We call the step where test accuracy reaches 0.95 the grokking step T. The rest of our project is about trying to predict T.

### Slide 3. The task is addition on a clock with 23 hours (0:25, total 1:20)

Say: The task is modular addition with modulus 23, like a clock with 23 hours. Every one of the 529 pairs is in the dataset. Each run trains on 476 pairs and must fill in the other 53. The network is a two-layer perceptron, and this is the baseline that Kumar and coauthors used. It groks even without weight decay.

### Slide 4. What sets the length of the delay is disputed (0:20, total 1:40)

Say: The literature disagrees about what sets this delay. The current candidates are the weight norm, the weight-decay rate, and a move from lazy to rich training, where the network's kernel starts to adapt. To decide between them we need to measure the kernel itself.

## Turn 2. The kernel, lazy and rich, growth and shape, the three numbers (2:10)

### Act 2. The tangent kernel (0:05, total 1:45)

Say: What does it mean for a network to learn features, and how could we see it happen?

### Slide 5. The tangent kernel says how a step on one input moves the output on another (0:40, total 2:25)

Say: Suppose we take one gradient step to fix the output on one input, x. The weights move in the direction of the gradient at x. That also changes the output on every other input. For an input x-prime, the change is proportional to the inner product of the two gradients, which is the tangent kernel. If the kernel is large, the network treats the two inputs alike. Collecting these values over every pair of test inputs gives a matrix. It describes what the network currently thinks is similar.

### Slide 6. In the lazy regime the kernel stays fixed during training (0:25, total 2:50)

Say: For an infinitely wide network the kernel does not change, so training is a fixed kernel method. That is the lazy regime. Multiplying the output by a factor alpha pushes a network towards it. When the kernel does change, the network learns new features, which is the rich regime. Kumar and coauthors propose that grokking is the switch from lazy to rich.

### Slide 7. A kernel can grow, or it can change shape (0:25, total 3:15)

Say: Think of the kernel as an ellipse. Its axes say which directions in the data the network cares about. The kernel can grow, so the ellipse gets bigger but points the same way. Or it can change shape, so the ellipse turns or stretches. Only the second one changes what the network treats as similar. Both can produce the same distance from the start.

### Slide 8. Three numbers describe how the kernel moved (0:35, total 3:50)

Say: We measure three numbers on the kernel. Scale, in blue, is its size relative to the start. Rotation, in orange, is one minus the cosine between the kernel now and at the start. It ignores size and only sees shape. Alignment, in green, is how well the kernel lines up with the labels. These three colours stay the same in every figure for the rest of the talk.

## Turn 3. The identity, the theory, the predictions and the grid (1:45)

### Slide 9. The distance mixes growth and rotation (0:20, total 4:10)

Say: Because all three numbers come from one matrix, the squared distance splits into a scale part and a rotation part. If the kernel only grows, the rotation part is zero and the distance is just the change in size. So a large distance does not prove that the network learned new features. That is why we track scale and rotation separately.

### Slide 10. In theory, weight decay only rescales the kernel (0:25, total 4:35)

Say: Lewkowycz and Gur-Ari wrote down how the kernel changes under gradient flow with weight decay. The weight decay only enters a term that shrinks the kernel evenly. Any change of shape has to come from the second term, T, which starts out small for wide networks. So if rotation sets the clock, and T stays small during training, wider networks should grok later. The weight decay should then matter much less. That second step is our own reading.

### Act 3. The experiment (0:05, total 4:40)

Say: How can one experiment tell the three clocks apart?

### Slide 11. Each account predicts a different result (0:30, total 5:10)

Say: Each account makes different predictions. A rotation clock says grokking happens when rotation reaches a fixed value. It also says that wider networks grok later and that weight decay barely matters. A weight-decay clock says the time is proportional to one over the weight decay. A norm clock says the scale is fixed at grokking. We designed one test for each column of this table. In the last column we test only the kernel statistics. We set the weight decay ourselves, so we know it exactly. In a general network the effective regularisation cannot be measured directly, while the tangent kernel can.

### Slide 12. We trained 720 networks across laziness, width and weight decay (0:25, total 5:35)

Say: We trained 720 networks. The output scale alpha controls laziness. The width controls how fast the kernel can rotate. And weight decay is the knob of the weight-decay clock. Every combination has five seeds. Runs that never grok are kept as censored data, so the slow cells are not thrown away.

If asked: The grid has 144 cells of 5 seeds, and each run is 200,000 steps. The runs took 18.9 hours of compute in all.

## Turn 4. The baseline, Tier 1 and Tier 2 (2:20)

### Act 4. Results (0:05, total 5:40)

Say: Which clock does the grokking time follow?

### Slide 13. Baseline: the network memorises early, groks late, and its kernel mostly grows (0:30, total 6:10)

Say: These are the five baseline runs at alpha 1, with the median as the line. They memorise at a median step of 2,983 and grok at a median step of 23,000. Look at the kernel. By the end the scale is 4.5 times its starting value, while the rotation is only 0.074. At grokking, 90% of the squared distance is growth. As Kumar and coauthors found, lazier networks grok later. At alpha 2 no seed groks within 200,000 steps.

### Slide 14. Tier 1: at a fixed width, rotation is the statistic most tied to grokking (0:45, total 6:55)

Say: Suppose grokking happens when a statistic reaches a threshold. Then that statistic should take nearly the same value at every run's grokking step. But a statistic that barely moves would pass that check too. So we also read each run at somebody else's grokking step. A real threshold gets much looser when we do that. Rotation tightens the most, by a factor of 1.71. Alignment barely tightens at all.

If asked: The test uses the 78 runs of width 100 that grok, with 2,000 shuffles. The ratio is 1.71 for rotation, 1.29 for scale and 1.09 for alignment. No shuffle gave rotation or scale a spread as small as the true one.

### Slide 15. Reaching that rotation is not enough to grok (0:20, total 7:15)

Say: The tie only holds within one width. Wider networks grok with less rotation but take longer. This suggests that width sets how fast the kernel rotates. And some networks reach the usual rotation without ever grokking, so rotation alone does not make a network grok.

If asked: At grokking, rotation falls from 0.108 at width 50 to 0.019 at width 1600. Of the 34 cells in which a seed memorises but none groks, 8 have a seed that still reaches the median rotation at grokking of its width. The alignment barely moves. Its median is 0.398 at the start and 0.406 at grokking.

### Slide 16. Tier 2: the grokking time depends on both width and weight decay (0:40, total 7:55)

Say: We fit how the grokking time scales with width and with weight decay. A rotation clock predicts a positive width exponent and no decay effect. A weight-decay clock predicts no width effect and an exponent of minus one. We find both. The width exponent a is plus 0.24 and the decay exponent b is minus 0.63. Both intervals exclude zero, and neither value is what its account predicts on its own. This holds for the range of weight decay where every seed groks.

If asked, or if there is time, read the figure: Each coloured line is one width, and each point on it is the median grokking step at one weight decay. Faint dots are single seeds, and open circles at the top are runs that never grokked. The dashed lines are our fit. The two grey lines at the lower right show the slope each clock predicts. A rotation clock gives a flat line, and a weight-decay clock gives a slope of minus one. Between ten to the minus five and ten to the minus four, where every seed groks, our lines fall, but on average less steeply than minus one. At ten to the minus five the wider networks sit higher. Adding the next decay level, three times ten to the minus four, moves b towards zero. The fit uses the 90 runs at alpha 1 in that range.

## Turn 5. Tier 3, MNIST, the limitations and the conclusion (2:05)

### Slide 17. Tier 3: rotation is the best single predictor, and scale adds what rotation lacks (0:40, total 8:35)

Say: Now we ask which statistic, measured early, tells us when a run will grok. We use only the kernel statistics here, because they can be measured in any network. We read the three numbers at step 2,430, before any run has grokked. Then we predict the grokking step of cells the model has never seen. Each bar appears on a click. Alignment alone explains almost nothing. Scale explains more, and rotation explains the most. Adding scale to rotation raises the held-out R squared from 0.35 to 0.51. Adding alignment on top only takes it to 0.55.

If asked: The model is kernel ridge regression on 355 runs that grok, with each of the 74 cells held out in turn.

### Slide 18. Tier 4: on MNIST the time is set by weight decay alone (0:30, total 9:05)

Say: We asked the same two questions on MNIST, in the setup of Omnigrok, and got a cleaner answer. The grokking time is one over the weight decay: the decay exponent is minus 1.00, and the width exponent is minus 0.02, so width does not enter. When a network groks, its weight norm always falls to the same value, 2.11 times its starting value, within 3% across every scale, width and decay. Weight decay shrinks the norm at a fixed rate, so the time is how long that takes. We did not derive that law. It comes from Liu and coauthors. Rotation falls with width, from 0.045 at width 200 to 0.002 at width 800, but the time does not move.

If asked: The grid is 560 runs: 4 initial scales, 4 widths, 7 weight decays and 5 seeds, each run 500,000 steps. It uses 1,000 training images and all 10,000 test images. The network is a three-layer rectifier network without biases on the squared error. The grokking level is 0.80, because no run reaches 0.95 within the budget. The fit uses initial scale 7 at widths 200 to 800, where every seed groks, and the exponents are the same at scales 5 and 9. Width 100 is left out, because its test accuracy at memorisation is already 0.72, so there is little delay to time. We read that as a limit of capacity at 1,000 images. Without weight decay no run reaches the level. The learning rate is reset from the curvature every 10,000 steps, which sets the memorisation step. The placebo test cannot separate the statistics here, because each is a function of the step and the grokking times span more than a decade. We did not repeat Tier 3, because a residual of 0.03 makes the time a function of the settings alone. Why the two tasks differ is our own reading. On modular addition a fixed kernel cannot solve the task, as Mohamadi and coauthors show, so the kernel has to change shape first.

### Act 5. What we found (0:05, total 9:10)

Say: So what sets the grokking time?

### Slide 19. Limitations (0:15, total 9:25)

Say: We used one task and one network family for the main grid, so we cannot claim the picture holds in general. The exponents depend on the range of weight decays. The kernel statistics are not a better forecast than the settings themselves. And MNIST changes three things at once. So it tells us which clock sets the time once a delay exists, and not whether rotation can cause one.

If asked: Each cell has 5 seeds, and each event is known only to the nearest checkpoint. The kernel-flow result assumes a homogeneous network, which the centred predictor is not. On MNIST the learning rate is reset from the curvature every 10,000 steps, which sets the memorisation step. Whether the group accepts that rule is still open. Width 100 does not follow the MNIST law and is left out of the fits. The alignment uses the kernel of the summed logits, and the kernel of all p logits may behave differently.

### Slide 20. Weight decay sets the time, through the weight norm (0:30, total 9:55)

Say: The usual kernel distance mostly records growth, so split it before you read it as feature learning. On modular addition, at a fixed width, rotation is the statistic most tied to grokking, and read early it is the best single predictor. Reaching that rotation does not make a network grok, and scale adds what rotation lacks. Across both tasks, the thing that sets the time is weight decay, acting through the weight norm. Width enters only on modular addition. Our reading of why is that a fixed kernel cannot solve that task, so the kernel has to change shape first.

### Slide 21. Thank you (0:05, total 10:00)

Say: Thank you. We are happy to take questions.
