# CLAUDE.md

This file sets the rules for any assistant working in this repository. Read it before doing anything else. The rules below override default behaviour.

## What this project is

This is the STAT3007 deep learning project for Group G2. The proposal is in `DeepLearningProposal.pdf` at the repository root. Read it before starting work on the code or the report.

The project asks what sets the grokking time on modular addition. The candidates are rotation of the empirical neural tangent kernel, the scale of the kernel or the weights, or the weight-decay rate itself. The plan is to first reproduce a published baseline, then extend it with new measurements and a new test.

The baseline is the setup in Kumar et al. (2024), Appendix 8.3. It is a one hidden layer MLP with width 100, input size 2p as two concatenated one-hot vectors, output size p, mean squared error loss, full-batch gradient descent with learning rate 100, p equal to 23, and 90 percent of the p squared pairs used for training. The proposal calls this Tier 0 and treats it as a hard gate. If Tier 0 does not match the published figure, later work is not interpretable.

The extension measures three quantities on a fixed probe set at each checkpoint. The scale term S_t is the log ratio of the Frobenius norm of the kernel at time t to the norm at time 0. The rotation term R_t is one minus the Frobenius inner product of the two normalised kernels. The centred kernel-target alignment A_t follows Cortes, Mohri, and Rostamizadeh (2012) and is unchanged by rescaling. The head claim is a regression of log grokking time on log width and log of the product of learning rate and weight decay.

## Rule 1. Never invent a source

You must not fabricate a citation, a paper title, an author list, an arXiv identifier, a theorem number, an equation number, or a page number. If you cannot verify a source, say so instead of guessing.

Before you cite a paper, check it against arXiv. Fetch the abstract page for the identifier and confirm that the title, the authors, and the year match what you plan to write. If a claim points at a specific theorem, appendix, or equation, open the paper and confirm the number. Do not cite from memory.

If a citation cannot be verified, write it with a clear marker such as `[UNVERIFIED]` and tell the user. Do not silently drop it and do not silently replace it with a source you think is close.

The references below are the ones in the proposal. Their arXiv identifiers were checked by the group on 31 August 2026. The venue lines for Jacot (2018), Nanda (2023), and Power (2022) were not checked at that time, so treat those venue lines as unverified until someone confirms them.

- Chizat, Oyallon, and Bach (2019), lazy training. arXiv:1812.07956.
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

Adding a paper to this list does not make it verified. Each new entry must go through the same check.

## Rule 2. Write in plain, full sentences

Write in complete sentences that read naturally. Each sentence should make one point. Keep the vocabulary simple and choose the common word over the rare one.

Avoid adjectives and adverbs that do not add information. Do not use hyperbole. Words such as "groundbreaking", "remarkable", "elegant", "powerful", and "crucial" are almost never needed. Say what a thing is and what it does.

Prefer prose over fragments. A list is fine when the items are truly parallel, but each item should still be a sentence. Do not write in note form or use arrows and colons in place of verbs.

This rule applies to chat replies, code comments, docstrings, commit messages, the report, and the LaTeX source.

## Rule 3. Cross-check anything taken from a source paper

This project reproduces published work and then extends it. When you implement or describe something that comes from one of the source papers, open that paper and check that what you wrote matches it.

This means the following. When you write the baseline, compare each hyperparameter against Kumar et al. (2024), Appendix 8.3. When you write the kernel decay result, compare it against Lewkowycz and Gur-Ari (2020), Theorems 1 and 2 and Equation S5. When you write the alignment measure, compare it against Cortes, Mohri, and Rostamizadeh (2012). When you quote a number from Truong et al. (2026), compare it against their paper. The proposal records some checked numbers on its fourth page and you may use those as a starting point, but the paper is the authority.

If your result or your reading disagrees with the paper, say so plainly and show both values. Do not adjust your result to match the paper and do not adjust your reading of the paper to match your result.

When something in the proposal turns out to differ from the source, the source wins and the proposal should be corrected. One checked case: the proposal body cites Kumar et al. Appendix 13 for the weight-decay argument, and the draft notes on page four say it should be 14. Both the arXiv v3 text and the ICLR 2024 camera-ready number the weight-decay section 13, so the body is right and the note is wrong. Cite the arXiv version number when you cite a section number, because numbering can change between versions.

## Rule 4. LaTeX must be verified and must compile

Rules 1 to 3 apply to LaTeX just as they apply to prose. Every citation in a `.tex` file must be verified, every sentence must be plain and complete, and every claim taken from a source paper must be cross-checked.

In addition, any LaTeX you write or edit must compile. After you change a `.tex` file, build it and confirm the build finishes without errors. The compiler on this machine is Tectonic. Build with the command below from the directory that holds the main file.

```
tectonic main.tex
```

Replace `main.tex` with the actual entry file. If the document uses a bibliography, make sure the `.bib` entries match the verified references above and that every `\cite` key exists. Read the log and fix undefined references, missing citations, and overfull boxes that change the page count. Do not report a document as done if it has not been built.

Do not change the template settings that the brief says not to change. The proposal notes record that the font size, page size, and the marked "don't change" lines must stay as they are.

## Working conventions

Use Python 3 for experiments. Use `uv` to manage the environment where possible.

Keep experiments reproducible. Fix seeds, record every hyperparameter with each run, and save outputs in a form that can be reloaded. The proposal pre-registers the grokking time as the step gap between the training loss crossing one threshold and the test loss crossing another. Cells that never grok are right-censored and kept, not dropped. Do not change these definitions without telling the user.

When you report an experimental result, state what was run, what was measured, and what the numbers were. If a run failed or was skipped, say so.

Do not add co-author trailers or generated-by footers to commits.
