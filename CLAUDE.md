# CLAUDE.md

This file sets the rules for any assistant working in this repository. Read it before doing anything else. The rules below override default behaviour.

## What this project is

This is the STAT3007 deep learning project for Group G2. The title is "Scale or Shape? Decomposing NTK Movement During Grokking". There are two documents. The proposal is `proposal.tex` and is finished apart from small edits. The report is `report.tex` and is still a scaffold. Read both before starting work on the code or the write-up, because they hold the full statement of the argument and the experimental design.

The project asks what sets the grokking time on modular addition. The usual evidence that grokking is a lazy-to-rich transition is the relative kernel movement, the Frobenius norm of the difference between the kernel at time t and the kernel at time 0 divided by the norm at time 0, and the project's core observation is that this statistic is not identifiable. It mixes a change in kernel scale with a change in kernel shape, because

    ||K_t - K_0||_F^2 / ||K_0||_F^2 = exp(2 S_t) + 1 - 2 exp(S_t) (1 - R_t).

The three measured terms are the scale term `S_t`, the log ratio of the Frobenius norms of the kernel at time t and at time 0; the shape term `R_t`, one minus the Frobenius inner product of the two normalised kernels; and the centred kernel-target alignment `A_t`, which follows Cortes, Mohri, and Rostamizadeh (2012) and is unchanged by rescaling. All three are computed on the centred kernel `H K H`, and an uncentred alignment is also recorded for comparison with Kumar et al.

The hypothesis H is that shape change is the clock. It says that generalisation begins when `A_t` crosses a threshold `A_*`, that `A_*` is shared across the laziness parameter alpha, the weight decay `eta*lambda`, the width `N` and the modulus `p`, and that the grokking time is independent of `S_t` given `A_t`. The head claim is the regression of log grokking time on log width and log of the product of learning rate and weight decay. Under H the width exponent is positive and the decay exponent is about zero. Under a weight-decay clock the width exponent is about zero and the decay exponent is about minus one. The report also fits a reciprocal form that admits the zero-decay column, which the log fit cannot.

The work is organised into tiers, and each tier is meant to be reportable on its own. Tier 0 reproduces the baseline of Kumar et al. (2024), Appendix 8.3, and is a hard gate. Tier 1 measures the three terms and the grokking time over the alpha by weight-decay grid and tests whether `A_*` is invariant. Tier 2 is the head claim. Tier 3 is a stretch replication of the clamped-norm law under mean squared error. Tier 4 is droppable theory.

The baseline is one hidden layer of width 100, input size 2p as two concatenated one-hot vectors, output size p, mean squared error loss, full-batch gradient descent with learning rate 100, p equal to 23, 90 percent of the p squared pairs used for training, and no weight decay. If Tier 0 does not match the published figure, later work is not interpretable.

## Repository layout

- `proposal.tex` is the proposal source. It replaces the earlier `DeepLearningProposal.pdf` and `G2_proposal.tex`, both of which were deleted.
- `report.tex` is the report source. Most of its body is still bullet lists written as prompts for prose, wrapped in a scaffold comment and marked with a `\TODO` macro.
- `ref.bib` holds the bibliography for both documents.
- `repoduced-code/` holds the experiments. The folder name is misspelled in the repository and should be left alone unless the user asks for it to be renamed.
- `repoduced-code/RepoducedCode.py` holds the dataset, the mean-field model, the empirical kernel, the raw scale and shape terms, and the alignment.
- `repoduced-code/ntk_lib.py` builds on that module with both parameterisations, the training loop, the grokking time, the censored fit, and the plot helpers. It is the module the notebooks import.
- `repoduced-code/tier0_reproduction.ipynb` and `repoduced-code/tier1_alignment_threshold.ipynb` are the two finished notebooks. The three `sam_*.ipynb` notebooks are earlier exploratory work.
- `repoduced-code/results/` holds one JSON file per run and a compressed kernel file beside it. Runs are cached, so `run_or_load` reloads a saved run when its configuration matches and retrains it otherwise.
- `pyproject.toml` and `uv.lock` define the environment.

## Rule 1. Never invent a source

You must not fabricate a citation, a paper title, an author list, an arXiv identifier, a theorem number, an equation number, or a page number. If you cannot verify a source, say so instead of guessing.

Before you cite a paper, check it against arXiv. Fetch the abstract page for the identifier and confirm that the title, the authors, and the year match what you plan to write. If a claim points at a specific theorem, appendix, or equation, open the paper and confirm the number. Do not cite from memory.

If a citation cannot be verified, write it with a clear marker such as `[UNVERIFIED]` and tell the user. Do not silently drop it and do not silently replace it with a source you think is close.

The entries below are the ones in `ref.bib`. The first group was checked by the group against the arXiv abstract pages on 4 September 2026, and the venue lines for Jacot, Chizat, Lewkowycz, Nanda, and Power were rechecked on the same day.

- Atanasov, Bordelon, and Pehlevan, silent alignment. arXiv:2111.00034.
- Chizat, Oyallon, and Bach, lazy training. arXiv:1812.07956.
- Cortes, Mohri, and Rostamizadeh (2012), centred alignment. JMLR 13, pages 795 to 828.
- Gromov (2023), grokking modular arithmetic. arXiv:2301.02679.
- Jacot, Gabriel, and Hongler (2018), neural tangent kernel. arXiv:1806.07572.
- Kim (2026), weight-decay clock. arXiv:2607.23967.
- Kumar et al. (2024), lazy to rich transition. arXiv:2310.06110.
- Lewkowycz and Gur-Ari (2020), training dynamics with L2 regularisation. arXiv:2006.08643.
- Mohamadi et al. (2024), theory of grokking modular addition. arXiv:2407.12332.
- Nanda et al. (2023), progress measures for grokking. arXiv:2301.05217.
- Power et al. (2022), grokking. arXiv:2201.02177.
- Truong (2026), logit-scale mediation. arXiv:2606.18465.
- Truong et al. (2026), weight norm delay law. arXiv:2606.13753.

Three further entries are cited but not yet confirmed by anyone in the group, so treat them as unverified until someone checks them.

- Liu, Michaud, and Tegmark, Omnigrok. arXiv:2210.01117. Cited in the related work of `report.tex`, with a note asking whether to keep it.
- Tian, scaling laws of feature emergence. arXiv:2509.21519. Cited in one line of `report.tex`, with a note asking whether to cut it.
- Kornblith et al., centred kernel alignment, cited in `report.tex` under the key `kornblith2019cka`. There is no entry for this key in `ref.bib`, so the report does not yet resolve it. Verify the paper before adding an entry, and do not write an identifier for it from memory.

Adding a paper to any list above does not make it verified. Each new entry must go through the same check.

Two conventions in `ref.bib` are easy to misread. The citation key carries the venue year while the `year` field carries the arXiv posting year, so `kumar2024grokking` has `year = {2023}`. Do not make one match the other without checking which is right. The `truong2026norm` and `truong2026logit` entries give the family name first in a Vietnamese name order, and this was how the group entered them.

## Rule 2. Write in plain, full sentences

Write in complete sentences that read naturally. Each sentence should make one point. Keep the vocabulary simple and choose the common word over the rare one.

Avoid adjectives and adverbs that do not add information. Do not use hyperbole. Words such as "groundbreaking", "remarkable", "elegant", "powerful", and "crucial" are almost never needed. Say what a thing is and what it does.

Prefer prose over fragments. A list is fine when the items are truly parallel, but each item should still be a sentence. Do not write in note form or use arrows and colons in place of verbs.

This rule applies to chat replies, code comments, docstrings, commit messages, the report, and the LaTeX source.

The bullet lists in `report.tex` are the one exception, and only because they are scaffolding. Every one of them is a writing prompt that has to be replaced by paragraphs. When you fill in a section, turn its list into prose rather than tidying the list. The scaffold comment near the top, the `\TODO` markers, and the red `\TODO` macro all have to be gone before submission.

## Rule 3. Cross-check anything taken from a source paper

This project reproduces published work and then extends it. When you implement or describe something that comes from one of the source papers, open that paper and check that what you wrote matches it.

This means the following. When you write the baseline, compare each hyperparameter against Kumar et al. (2024), Appendix 8.3, and compare the centred and rescaled predictor against their Appendix 8.1, Equation 7. When you write the kernel decay result, compare it against Lewkowycz and Gur-Ari (2020), Theorems 1 and 2 and Equation S5, whose numbering in arXiv v1 becomes S8 in v2, so cite the version. When you write the alignment measure, compare it against Cortes, Mohri, and Rostamizadeh (2012), Lemma 1. When you quote a number from Truong et al. (2026) or Truong (2026), compare it against their papers.

If your result or your reading disagrees with the paper, say so plainly and show both values. Do not adjust your result to match the paper and do not adjust your reading of the paper to match your result.

When something in the proposal turns out to differ from the source, the source wins and the proposal should be corrected. One checked case: the weight-decay argument is in Kumar et al., Appendix 13, in both the arXiv v3 text and the ICLR 2024 camera-ready, and Appendix 14 is momentum. Cite the arXiv version number when you cite a section number, because numbering can change between versions.

One cross-check is still open and is marked in `report.tex`. Kumar et al., Section 5, report a kernel-target alignment and name it after Kornblith et al., whose measure is centred, but the formula they print is not centred. Check the printed form against the PDF before resting any comparison on it.

## Rule 4. LaTeX must be verified and must compile

Rules 1 to 3 apply to LaTeX just as they apply to prose. Every citation in a `.tex` file must be verified, every sentence must be plain and complete, and every claim taken from a source paper must be cross-checked.

In addition, any LaTeX you write or edit must compile. There are two entry files, `proposal.tex` and `report.tex`, and each is built on its own from the repository root.

```
tectonic proposal.tex
tectonic report.tex
```

Tectonic is the compiler the group uses, but as of 21 September 2026 no `tectonic` binary is on the PATH on this machine and no other TeX installation is visible. If the build cannot be run, say so plainly and do not report the document as done. Do not silently substitute a different compiler.

Both documents read `ref.bib` through biblatex with the bibtex backend, so a build needs the bibliography pass as well. Make sure every `\cite` key exists in `ref.bib`. At present `report.tex` cites `kornblith2019cka`, which does not, so that citation will not resolve. Read the log and fix undefined references, missing citations, and overfull boxes that change the page count.

Do not change the template settings that the brief says not to change. The font size, the page size, and every line marked "Don't change this" must stay as they are. The proposal body has to stay within two pages with the references starting on their own page, and its preamble has extra spacing settings that exist to keep it there, including `\raggedbottom` and a shrink-free `\topsep` that stops a small overrun from collapsing the title block onto the author line. The report body has to stay within twelve pages excluding references and appendices, and its appendices within five pages. Submit the proposal as `G2_proposal_<SubmitterName>.pdf`.

## Working conventions for the experiments

Use Python 3.11 or later. Use `uv` to manage the environment. The runs are small enough for the CPU default in `ntk_lib.py`.

Keep experiments reproducible. Fix seeds, record every hyperparameter with each run, and save outputs in a form that can be reloaded. The model seed and the data seed are separate. A run is saved as one JSON file in `repoduced-code/results/` named after its cell, with the probe kernels in a compressed file beside it, and `run_or_load` retrains only when the saved configuration does not match the requested one. Checkpoints are recorded on a step grid, and any new sweep should keep the log-spaced reading of the results in mind, because grokking spans several orders of magnitude in step count.

A few names differ between the documents and the code. The weight decay that the write-up calls `eta*lambda` is the `eta_kappa` configuration key, and the code turns it into a PyTorch weight decay by dividing by the learning rate. The shape term that the write-up calls `R_t` was called the rotation term in earlier drafts. The history records both the raw terms `S_t` and `R_t` and their centred versions `S_c` and `R_c`, and both the centred alignment `A_t` and the uncentred one `A_u`.

The parameterisation is a real variable, not a detail. `ntk_lib.py` provides both a mean-field model and an NTK model, and the default in `DEFAULTS` is mean-field while every grid cell is run in NTK parameterisation. The report explains why the NTK parameterisation is required for the width exponent to mean anything and notes that Kumar et al. do not state which parameterisation their modular-addition network used. Whenever you report a width result, state which parameterisation produced it.

## Pre-registered definitions and the deviations recorded so far

The grokking time is the step gap between a memorisation event and a generalisation event. Cells that never generalise within the step budget are right-censored and kept, not dropped, and their grokking time is recorded only as a lower bound. Estimation is by censored regression rather than least squares on the survivors. Do not change these definitions without telling the user.

Three deviations from the proposal are already recorded in the notebooks, and they are deliberate. Keep them, and keep saying so in the write-up rather than quietly aligning the text to the proposal.

- The primary detection is on accuracy rather than loss. Memorisation is the first step at train accuracy 1, and generalisation is the first step at which test accuracy reaches the level set in the notebook. The loss definition is kept as a secondary, because the test loss settles between 0.01 and 0.03 long after accuracy is perfect and so rarely records a generalisation event within a run. `report.tex` carries a note that this has to be stated in the text.
- The Tier 0 alpha values follow Kumar et al., Figure 2(c), which sweeps 0.5, 1, 1.5 and 2, rather than the wider sweep written into the proposal.
- The Tier 1 weight-decay axis was refined downwards to 0, 1e-5, 3e-5, 1e-4 and 3e-4, because the proposal's values of 1e-3 and above shrink the weights within a few thousand steps.

When you report an experimental result, state what was run, what was measured, and what the numbers were. If a run failed or was skipped, say so.

Do not add co-author trailers or generated-by footers to commits.
