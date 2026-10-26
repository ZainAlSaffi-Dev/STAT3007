# Zain's script for Turn 5

This is the script for Turn 5 of the seminar, slides 19 to 24. Each section below is one paragraph to read aloud. Together they run to about two minutes at the deck's pace of 150 words a minute. Every number comes from the report's `numbers.tex`, through `_variables.yml`. The ladder of models on slide 19 comes from `model_fitting/4_nonlinear_exploration.ipynb`, Part 3, and the script gives no number for it.

## The script

### Slide 19. Tier 3: predicting the grokking time (about 1:00)

Last, can the kernel, read early, tell us when a run will grok? We read scale, shape and alignment at step 2,430, before any run groks, and predict the grokking time on cells the model has never seen. We climbed a ladder of models, from straight lines up to kernel ridge regression, which can fit any smooth surface. Two simple bends did most of the work: the log of shape, and a curve in scale. Almost half the runs have R below 0.01, so the log spreads them out, and with it all three statistics reach an R squared of 0.55 in place of 0.47. Alignment explains almost nothing. Shape alone explains the most, at 0.35, and adding scale lifts it to 0.51. So shape is the best single reading, but it needs scale, which is the mixed clock. The settings still predict better, but any network's kernel can be measured.

### Slide 20. Tier 4: MNIST (about 0:25)

We then repeated the tests on MNIST digits in the Omnigrok setup. There the starting weights are large, and weight decay has to shrink them. The time goes as one over the weight decay, width drops out, and the weight norm at grokking is the same within 3%. Our reading is that MNIST never forces the kernel to change shape.

### Slides 21 and 22. Limitations (about 0:12)

We used one task and the NTK parameterisation, where wider networks stay lazier. In 70 of the 144 cells no seed groks, and MNIST changes three things at once.

### Slides 23 and 24. What we found (about 0:28)

So what sets the grokking time? The usual kernel distance mostly records growth, so split it into scale and shape first. Shape is the most informative statistic, but it is not enough alone and it is not a fixed threshold. No single account holds, ours included. Width acts through shape, weight decay acts through scale, and both set the time. Thank you, and we are happy to take questions.

## Questions

These are not part of the timed talk.

- **Why the log of R?** We fitted R as it is and log R in straight lines and in kernel ridge, and log R predicted better on held-out cells. A straight line in shape alone explains 0.29 of log T with R as it is and 0.37 with log R.
- **Did you try the log of S?** No. S entered on its own scale, and a square term and splines let its curve bend.
- **Why hold out whole cells?** The five seeds of a cell are close copies. Between 0.96 and 0.99 of the spread of log T, S and log R lies between cells, so the statistics say nothing about which seed of a cell groks first.
- **Does the kernel beat weight decay?** No. On an unseen alpha, trees on width and weight decay reach a concordance of 0.79, and weight decay alone reaches 0.72. Trees on the three statistics reach 0.70, and a recurrent network on their history reaches 0.76.
- **What is kernel ridge regression?** It predicts a run's time from the runs that look most like it, with a Gaussian kernel that sets what "look like" means. That lets it follow curves and interactions without us choosing their form.
