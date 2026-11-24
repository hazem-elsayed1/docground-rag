# Development question draft for human review

All 24 items are provisional pending repository-owner approval. The reported independent semantic review cleared D20 and D22. Corpus scope is the four canonical scikit-learn 1.9.1 documents in [the manifest](../corpus/scikit-learn-1.9.1-v1.json). Corpus revision is `01021d79036af9e2a1da8a58ab50e354c4cc3197bfa4eaa55c96e45351f59dca`. Evidence comes directly from `data/processed/corpus-v1/documents.json`, SHA-256 `83898f9410710eb4f7acf3b97d0602235687a86ed7ededa0fb002bcca75edfab`.

IDs D01–D24 track this review draft. The proposed family slugs describe the intent and can persist through wording revisions. Each lettered claim is a proposed required evidence unit. Document names below are manifest IDs. Anchors and quotations locate the candidates for review. Coordinate export follows semantic approval under [annotation schema v2](../README.md#multi-document-annotations-and-evaluation).

## Coverage

The table counts each answerable question once under its primary document. Four unanswerable items apply to the entire corpus. Allocation follows the approved category quotas and the available distinct intents.

| Category | Common Pitfalls | Compose | Cross-validation | Model Evaluation | Corpus-wide | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct factual | 0 | 2 | 2 | 2 | 0 | 6 |
| Conceptual or explanatory | 1 | 1 | 2 | 2 | 0 | 6 |
| Multi-evidence within one document | 0 | 1 | 2 | 1 | 0 | 4 |
| Cross-document | 1 | 2 | 1 | 0 | 0 | 4 |
| Unanswerable | 0 | 0 | 0 | 0 | 4 | 4 |
| **Total** | **2** | **6** | **7** | **5** | **4** | **24** |

By primary document, the 20 answerable items are allocated as follows. Common Pitfalls has 2, Compose has 6, Cross-validation has 7, and Model Evaluation has 5. Each item contributes once to these counts. D17–D20 have four distinct primary document pairs. There are five question-and-pair combinations because D20 can also use the Model Evaluation and Cross-validation pair used by D19.

## Direct factual

### D01 · `pipeline-nested-parameter-key`

**Question.** A pipeline step named `clf` is an SVC. What key would you pass to `set_params` to set its `C` to 10?

**Expected answer and claim A.** Use `clf__C=10`. The example gives the exact call, so one unit suffices.

**Candidate A** · `compose` §8.1.1.1.4 Access to nested parameters · `#access-to-nested-parameters`

> ```pycon
> >>> pipe.set_params(clf__C=10)
> ```

### D02 · `estimator-repr-text-mode`

**Question.** Which setting switches scikit-learn’s notebook estimator display from its HTML representation to text?

**Expected answer and claim A.** Call `set_config(display='text')` from scikit-learn. The executable example supplies the option and value.

**Candidate A** · `compose` §8.1.5 Visualizing Composite Estimators · `#visualizing-composite-estimators`

> ```pycon
> >>> from sklearn import set_config
> >>> set_config(display='text')
> ```

### D03 · `leave-one-out-split-count`

**Question.** With 12 samples, how many train/test splits does `LeaveOneOut` produce?

**Expected answer and claim A.** Twelve. The general rule gives one split per sample.

**Candidate A** · `cross_validation` §3.1.2.1.3 Leave One Out · `#leave-one-out-loo`

> Thus, for \(n\) samples, we have \(n\) different training sets and \(n\) different test sets.

### D04 · `cross-validate-return-indices`

**Question.** Which `cross_validate` option makes the result retain the training and test indices for each split?

**Expected answer and claim A.** Set `return_indices=True`. The selected sentence names both the option and the returned information.

**Candidate A** · `cross_validation` §3.1.1.1 The cross_validate function and multiple metric evaluation · `#the-cross-validate-function-and-multiple-metric-evaluation`

> Similarly, you may set return_indices=True to retain the training and testing indices used to split the dataset into train and test sets for each cv split.

### D05 · `confusion-matrix-axis-convention`

**Question.** In scikit-learn’s confusion matrix, which axis is the true class and which is the predicted class?

**Expected answer and claim A.** Rows are true classes and columns are predicted classes. The first paragraph states the true-row convention. The predicted-column convention follows from the next paragraph, which defines entry `(i, j)` as the count for actual class `i` and predicted class `j`. Both passages support one evidence unit.

**Candidate A** · `model_evaluation` §3.4.4.6 Confusion matrix · `#confusion-matrix`

> The confusion_matrix function evaluates classification accuracy by computing the confusion matrix with each row corresponding to the true class (Wikipedia and other references may use different convention for axes).
>
> By definition, entry \(i, j\) in a confusion matrix is the number of observations actually in group \(i\), but predicted to be in group \(j\).

### D06 · `top-k-accuracy-multilabel-support`

**Question.** Does `top_k_accuracy_score` support multilabel classification?

**Expected answer and claim A.** It supports binary and multiclass classification. Multilabel classification is outside its stated scope. The full sentence settles the support question.

**Candidate A** · `model_evaluation` §3.4.4.3 Top-k accuracy score · `#top-k-accuracy-score`

> The function covers the binary and multiclass classification cases but not the multilabel case.

## Conceptual or explanatory

### D07 · `random-state-none-global-rng`

**Question.** Why does the Common Pitfalls chapter apply its advice about `RandomState` instances to `random_state=None` as well?

**Expected answer and claim A.** Passing `None` uses NumPy’s global `RandomState` instance. The note explicitly connects the two cases. The question asks about the identity of the RNG in use and has no requirement about seeds, folds, or process-level reproducibility.

**Candidate A** · `common_pitfalls` §12.3.1 Using None or RandomState instances, and repeated calls to fit and split · `#using-none-or-randomstate-instances-and-repeated-calls-to-fit-and-split`

> Since passing random_state=None is equivalent to passing the global RandomState instance from numpy (random_state=np.random.mtrand._rand), we will not explicitly mention None here. Everything that applies to instances also applies to using None.

### D08 · `feature-union-duplicate-features`

**Question.** If two `FeatureUnion` transformers produce the same feature, should you expect the union to remove the duplicate automatically? Who is responsible for avoiding it?

**Expected answer and claim A.** The caller must ensure disjoint features. `FeatureUnion` cannot detect identical output features. One passage states the behavior and its practical consequence.

**Candidate A** · `compose` §8.1.3 FeatureUnion: composite feature spaces · `#featureunion-composite-feature-spaces`

> (A FeatureUnion has no way of checking whether two transformers might produce identical features. It only produces a union when the feature sets are disjoint, and making sure they are is the caller’s responsibility.)

### D09 · `time-series-shuffle-autocorrelation`

**Question.** Why can shuffling time-ordered observations before CV make the validation score look better than it should?

**Expected answer and claim A.** Shuffling can place observations close in time on opposite sides of a split. This can make validation samples artificially similar to training samples and inflate the score. The chapter illustrates this with time-ordered articles.

**Candidate A** · `cross_validation` §3.1.3 A note on shuffling · `#a-note-on-shuffling`

> For example, if samples correspond to news articles, and are ordered by their time of publication, then shuffling the data will likely lead to a model that is overfit and an inflated validation score: it will be tested on samples that are artificially similar (close in time) to training samples.

### D10 · `stratification-fold-score-spread`

**Question.** Why can stratification make the spread of CV scores understate uncertainty when classes are rare?

**Expected answer and claim A.** More homogeneous folds hide some variability from fitting with limited observations, so the observed spread can shrink artificially. These adjacent statements in one note identify the mechanism and the consequence.

**Candidate A** · `cross_validation` §3.1.2.2 Cross-validation iterators with stratification based on class labels · `#cross-validation-iterators-with-stratification-based-on-class-labels`

> Stratification makes cross-validation folds more homogeneous, and as a result hides some of the variability inherent to fitting models with a limited number of observations.
>
> As a result, stratification can artificially shrink the spread of the metric measured across cross-validation iterations: the inter-fold variability does no longer reflect the uncertainty in the performance of classifiers in the presence of rare classes.

### D11 · `explained-variance-systematic-offset`

**Question.** If a regression model consistently predicts too high by a fixed amount, what distinction between explained variance and R² makes R² preferable?

**Expected answer and claim A.** Explained variance ignores systematic prediction offsets. The chapter gives this as the reason to prefer R² in general.

**Candidate A** · `model_evaluation` §3.4.6.8 Explained variance score · `#explained-variance-score`

> The difference between the explained variance score and the R² score, the coefficient of determination is that the explained variance score does not account for systematic offset in the prediction. For this reason, the R² score, the coefficient of determination should be preferred in general.

### D12 · `log-loss-probability-input`

**Question.** Why should you pass a classifier’s `predict_proba` output to log loss instead of its hard predictions?

**Expected answer and claim A.** Log loss is defined on probability estimates and evaluates probability outputs. The prose explicitly contrasts probability outputs with discrete predictions.

**Candidate A** · `model_evaluation` §3.4.4.12 Log loss · `#log-loss`

> Log loss, also called logistic regression loss or cross-entropy loss, is defined on probability estimates. It is commonly used in (multinomial) logistic regression and neural networks, as well as in some variants of expectation-maximization, and can be used to evaluate the probability outputs (predict_proba) of a classifier instead of its discrete predictions.

**Candidate alternative for A** · Probability-only scorer configuration · `model_evaluation` §3.4.3.2.2 Creating a custom scorer object · `#creating-a-custom-scorer-object`

> If the scoring function only accepts probability estimates (e.g. metrics.log_loss), then one needs to set the parameter response_method="predict_proba".

The alternative independently identifies log loss as accepting probability estimates and identifies `predict_proba` as the required scorer response. Verify the exact excerpt and final coordinates again at annotation export.

## Multi-evidence within one document

### D13 · `pipeline-cache-reuse-and-inspection`

**Question.** With pipeline caching enabled and PCA followed by SVC, when can two grid-search candidates reuse the same fitted PCA, and where should you inspect the fitted PCA in a fitted pipeline?

**Expected answer.** With caching enabled, the fitted non-final PCA step can be reused when its parameters and input data are identical. Inspect the fitted pipeline’s `named_steps` because caching clones the transformer before fitting.

**Candidate A** · Reuse conditions · `compose` §8.1.1.2 Caching transformers: avoid repeated computation · `#caching-transformers-avoid-repeated-computation`

> With its memory parameter set, Pipeline will cache each transformer after calling fit. This feature is used to avoid computing the fit transformers within a pipeline if the parameters and input data are identical. A typical example is the case of a grid search in which the transformers can be fitted only once and reused for each configuration. The last step will never be cached, even if it is a transformer.

**Candidate B** · Fitted instance inspection · same section

> Enabling caching triggers a clone of the transformers before fitting. Therefore, the transformer instance given to the pipeline cannot be inspected directly. In the following example, accessing the PCA instance pca2 will raise an AttributeError since pca2 will be an unfitted transformer. Instead, use the attribute named_steps to inspect estimators within the pipeline:

**Sufficiency.** A states the cache conditions for the non-final PCA step. B independently explains where to inspect the fitted clone. Both units are required.

### D14 · `cross-val-predict-partition-and-scoring`

**Question.** For an out-of-fold prediction plot, what must be true about test-set membership for `cross_val_predict`, and why can scoring those pooled predictions differ from averaging CV fold scores?

**Expected answer.** Each sample must be in a test set exactly once. Pooling combines predictions from distinct fitted models across samples, whereas `cross_val_score` scores by fold.

**Candidate A** · Test-set requirement · `cross_validation` §3.1.1.2 Obtaining predictions by cross-validation · `#obtaining-predictions-by-cross-validation`

> Only cross-validation strategies that assign all elements to a test set exactly once can be used (otherwise, an exception is raised).

**Candidate B** · Grouping and scoring distinction · same section

> The result of cross_val_predict may be different from those obtained using cross_val_score as the elements are grouped in different ways. The function cross_val_score takes an average over cross-validation folds, whereas cross_val_predict simply returns the labels (or probabilities) from several distinct models undistinguished.

**Sufficiency.** A establishes eligibility and B explains why those predictions cannot stand in automatically for fold-averaged scores. B allows coincident results for some metrics and fold designs.

### D15 · `stratified-group-fold-goal-and-limit`

**Question.** What two goals does `StratifiedGroupKFold` balance, and can it yield imbalanced folds even when perfect stratification is possible?

**Expected answer.** It attempts to preserve class proportions while keeping groups intact. It can still produce imbalanced splits even when perfect stratification is feasible.

**Candidate A** · Goals · `cross_validation` §3.1.2.4.2 StratifiedGroupKFold · `#stratifiedgroupkfold`

> StratifiedGroupKFold is a cross-validation scheme that combines both StratifiedKFold and GroupKFold. The idea is to try to preserve the distribution of classes in each split while keeping each group within a single split.

**Candidate B** · Limit · same section

> This split is suboptimal in a sense that it might produce imbalanced splits even if perfect stratification is possible.

**Sufficiency.** A and B establish the goals and the absence of a guarantee. If the answer must explain why, include the preceding algorithm paragraph. For the current wording the yes/no limit suffices, so the mechanism is optional context.

### D16 · `r2-negative-baseline-and-comparability`

**Question.** What does a negative R² say about performance relative to a constant mean predictor, and why should you be cautious comparing R² across different datasets? Assume the targets are nonconstant.

**Expected answer.** Negative R² is below the mean-prediction baseline of zero. Target variance depends on the dataset, which limits cross-dataset comparisons.

**Candidate A** · Baseline · `model_evaluation` §3.4.6.1 R² score, the coefficient of determination · `#r2-score-the-coefficient-of-determination`

> Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected (average) value of y, disregarding the input features, would get an \(R^2\) score of 0.0.

**Candidate B** · Dataset dependence · same section

> As such variance is dataset dependent, \(R^2\) may not be meaningfully comparable across different datasets.

**Sufficiency.** A gives the zero reference, and B independently gives the comparison caveat. The nonconstant target condition avoids the special handling described elsewhere in that section.

## Cross-document

### D17 · `mixed-text-categorical-fixed-validation`

**Question.** A classifier uses a `ColumnTransformer` with `CountVectorizer` on a DataFrame’s `title` field and `OneHotEncoder` on its `city` field. The dataset ships with fixed training and validation sets. How should you specify the two column selectors and encode the split for `PredefinedSplit`?

**Expected answer.** Select `title` using the string `'title'` and `city` using `['city']`. Set `test_fold` to 0 for validation rows and −1 for training rows.

**Candidate A** · Selector shapes · `compose` §8.1.4 ColumnTransformer for heterogeneous data · `#columntransformer-for-heterogeneous-data`

> In the above example, the CountVectorizer expects a 1D array as input and therefore the columns were specified as a string ('title'). However, OneHotEncoder as most of other transformers expects 2D data, therefore in that case you need to specify the column as a list of strings (['city']).

**Candidate B** · Fold encoding · `cross_validation` §3.1.2.3 Predefined fold-splits / Validation-sets · `#predefined-fold-splits-validation-sets`

> For example, when using a validation set, set the test_fold to 0 for all samples that are part of the validation set, and to -1 for all other samples.

**Sufficiency and necessity.** A covers the shapes for the exact transformers in the question. B supplies the fold markers. Compose gives no `PredefinedSplit` encoding. Cross-validation gives no selector shapes for those two transformers. Common Pitfalls and Model Evaluation supply neither combination.

**Candidate alternative for A** · The constructor independently shows both exact selectors · `compose` §8.1.4 ColumnTransformer for heterogeneous data · `#columntransformer-for-heterogeneous-data`

> ```pycon
> >>> column_trans = ColumnTransformer(
> ...     [('categories', OneHotEncoder(dtype='int'), ['city']),
> ...      ('title_bow', CountVectorizer(), 'title')],
> ...     remainder='drop', verbose_feature_names_out=False)
> ```

### D18 · `log-target-wrapper-and-growth-metric`

**Question.** For positive population counts that grow exponentially, which wrapper and forward and inverse functions let you fit a regressor on log-transformed targets and return predictions to the original scale? Which loss does the Model Evaluation chapter recommend for exponentially growing targets?

**Expected answer.** Configure `TransformedTargetRegressor` with a `func` that applies `np.log` to the positive targets and an `inverse_func` that applies `np.exp` to bring predictions back to the original scale. The chapter recommends mean squared logarithmic error (MSLE) for exponentially growing targets.

**Candidate A** · Target wrapper and output scale · `compose` §8.1.2 Transforming target in regression · `#transforming-target-in-regression`

> ```text
> For simple transformations, instead of a Transformer object, a pair of functions can be passed, defining the transformation and its inverse mapping:
>
> >>> def func(x):
> ...     return np.log(x)
> >>> def inverse_func(x):
> ...     return np.exp(x)
>
>
> Subsequently, the object is created as:
>
> >>> regr = TransformedTargetRegressor(regressor=regressor,
> ...                                   func=func,
> ...                                   inverse_func=inverse_func)
> >>> regr.fit(X_train, y_train)
> ```

**Candidate B** · Loss choice · `model_evaluation` §3.4.6.4 Mean squared logarithmic error · `#mean-squared-logarithmic-error`

> The mean_squared_log_error function computes a risk metric corresponding to the expected value of the squared logarithmic (quadratic) error or loss.
>
> If \(\hat{y}_i\) is the predicted value of the \(i\)-th sample, and \(y_i\) is the corresponding true value, then the mean squared logarithmic error (MSLE) estimated over \(n_{\text{samples}}\) is defined as
>
> \[\text{MSLE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2.\]
>
> Where \(\log_e (x)\) means the natural logarithm of \(x\). This metric is best to use when targets having exponential growth, such as population counts, average sales of a commodity over a span of years etc.

**Sufficiency and necessity.** A gives the exact log/exp pair, passes both functions by name, and fits the wrapper. The preceding introduction in the same section explicitly says that the inverse maps predictions back to the original space. B identifies MSLE and states the growth use case. Compose gives the wrapper example and reports R². Model Evaluation gives the MSLE recommendation. The corpus search found the wrapper configuration and recommendation only in their respective chapters. Keep the units separate. The target transform and the metric formula are distinct operations.

### D19 · `leave-one-experiment-out-balanced-score`

**Question.** You have multiple experiments, each with both binary labels present in unequal proportions. In its example of holding out one entire experiment at a time, which splitter does the Cross-validation chapter use? Which metric equals the arithmetic mean of sensitivity and specificity?

**Expected answer.** The example uses `LeaveOneGroupOut` with experiment IDs as groups. For binary classification, `balanced_accuracy_score` is the arithmetic mean of sensitivity and specificity.

**Candidate A** · Experiment holdout · `cross_validation` §3.1.2.4.3 Leave One Group Out · `#leave-one-group-out`

> For example, in the cases of multiple experiments, LeaveOneGroupOut can be used to create a cross-validation based on the different experiments: we create a training set using the samples of all the experiments except one:

**Candidate B** · Binary metric identity · `model_evaluation` §3.4.4.4 Balanced accuracy score · `#balanced-accuracy-score`

> In the binary case, balanced accuracy is equal to the arithmetic mean of sensitivity (true positive rate) and specificity (true negative rate), or the area under the ROC curve with binary predictions rather than scores:

**Sufficiency and necessity.** A names the splitter used in the example and identifies the held-out experiment. The same section identifies equivalent configurations of `LeavePGroupsOut` and `GroupKFold`. The question asks about the illustrated choice. B under the named metric heading gives the requested identity. The corpus search found the illustrated splitter in Cross-validation and the metric identity in Model Evaluation.

### D20 · `precision-recall-threshold-final-holdout`

**Primary document.** Common Pitfalls for the test-isolation decision.

**Question.** Which function lets you examine the precision–recall trade-off across decision thresholds, and can you use your final test set to select the threshold if you will report that set’s performance as a held-out estimate?

**Expected answer.** Use `precision_recall_curve` with labels and prediction scores on development or validation data. Common Pitfalls explicitly says to keep test data out of model choices, and Cross-validation explains how test-set tuning biases the estimate of generalization. Applying that guidance to the threshold requires an inference. Choosing the threshold selects part of the classifier’s decision rule. Choose it on development or validation data and reserve the final test set for the held-out estimate.

**Candidate A** · Threshold sweep · `model_evaluation` §3.4.4.9 Precision, recall and F-measures · `#precision-recall-and-f-measures`

> The precision_recall_curve computes a precision-recall curve from the ground truth label and a score given by the classifier by varying a decision threshold.

**Candidate B** · Final holdout isolation · `common_pitfalls` §12.2 Data leakage · `#data-leakage`

> Test data should never be used to make choices about the model.

**Candidate alternative for B** · Test-set leakage through parameter tuning · `cross_validation` §3.1 Cross-validation: evaluating estimator performance · `#cross-validation-evaluating-estimator-performance`

> When evaluating different settings (“hyperparameters”) for estimators, such as the C setting that must be manually set for an SVM, there is still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally. This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance.

**Inference.** A decision threshold determines which scores become positive predictions. Choosing it using final-test labels selects part of the classifier’s decision rule, so threshold selection is a model choice. The Cross-validation example uses estimator hyperparameters such as SVM `C`. Applying its leakage reasoning to a threshold requires the same inference as applying the Common Pitfalls rule.

**Sufficiency and necessity.** A identifies the curve function. Either B passage supports the test-isolation conclusion with the stated inference. There are two required units, with one location for A and two acceptable locations for B. Model Evaluation supplies A. Common Pitfalls and Cross-validation can each supply B, and neither describes the curve function. Model Evaluation §3.4.1 distinguishes prediction from decision-making, but gives no final-test isolation rule. Every complete evidence mapping therefore takes two documents.

## Corpus-scoped unanswerable

### D21 · `column-transformer-sparse-density-cutoff`

**Question.** According to the four pinned chapters, what is the default `sparse_threshold` value of `ColumnTransformer`?

**Missing evidence.** Compose shows an example calling `toarray()`, which establishes the representation for that example but provides neither the parameter default nor the density cutoff. An API reference outside the corpus would be needed for the value.

### D22 · `time-series-split-gap-semantics`

**Question.** According to the four pinned chapters, what does the `gap` parameter of `TimeSeriesSplit` control?

**Status.** The reported independent semantic review cleared this question as corpus-scoped unanswerable. Repository-owner approval is pending.

**Missing evidence.** The Time Series Split example prints `gap=0`. Its prose explains split ordering, accumulating training sets and comparable time durations. The default representation and example indices do not establish what changing `gap` controls. The four-chapter check found no explanation of the parameter.

### D23 · `log-loss-float32-probability-clipping`

**Question.** According to the four pinned chapters, what exact lower clipping bound does `log_loss` use for a predicted probability of zero when the probability array is float32?

**Missing evidence.** Model Evaluation gives the mathematical loss and examples with interior probabilities. Neither gives a zero-probability policy or dtype-specific numerical bound. The singularity of the formula cannot establish the implementation’s clipping behavior.

### D24 · `permutation-p-value-monte-carlo-interval`

**Question.** Which method do the four pinned chapters recommend for constructing a 95% Monte Carlo confidence interval around the `permutation_test_score` p-value from a finite number of permutations?

**Missing evidence.** Cross-validation explains how permuted scores produce a p-value and gives a guideline for the number of permutations. It gives no interval construction, and the recommendation cannot be inferred from the p-value estimate alone.

### Four-document absence checks

The recheck searched the canonical text and local HTML main text of all four documents for API names and related terms. It inspected sparse density defaults, temporal separation and exclusions for `TimeSeriesSplit`, float32 clipping, and confidence intervals around permutation p-values. The only literal `gap` occurrence in the four chapter bodies is the Time Series Split example’s printed `gap=0`. Reading both time-series sections and their examples yielded no parameter explanation. These checks cover the pinned chapters and their embedded examples.

| ID | Common Pitfalls | Compose | Cross-validation | Model Evaluation |
| --- | --- | --- | --- | --- |
| D21 | Preprocessing examples give no density policy. | §8.1.4 shows sparse example output and column handling without the cutoff. | Preprocessing in §3.1.1 concerns splitting and transformations. | Metric examples give no transformer stacking rule. |
| D22 | Splitter discussions in §12.3 concern randomness and repeat calls. | Composite-estimator CV examples give no time-series gap behavior. | §§3.1.2.6 and 3.1.2.6.1 explain temporal ordering and print `gap=0` without defining it. | Scoring and CV references give no time-series gap behavior. |
| D23 | Performance examples give no log-loss implementation details. | Target log transforms in §8.1.2 concern regression on transformed targets. | Scoring examples give no probability clipping rule. | §3.4.4.12 and the classification scoring references give no dtype-dependent bound. |
| D24 | Random-label example in §12.2.2 does not construct a permutation p-value interval. | Grid-search examples concern estimator configuration and validation. | §3.1.5 describes the p-value and permutation count guidance without an interval method. | Scoring and uncertainty-related terms give no interval recommendation for permutation testing. |

## Semantic-duplicate screen and adjudication

- **Pilot Q4 and Q6.** D17 asks about selector shapes and fixed split encoding. Pilot Q4 asks about fitting supervised feature selection inside a CV pipeline. D20 asks about a decision threshold and final-test use, distinct from fitting selectors or preprocessing statistics in pilot Q4/Q5. Human review retained D07 because it asks which RNG `None` uses. Pilot Q6 concerns integer seeds for two roles, and Q8 concerns repeatable runs with variation across folds.
- **Other pilot overlap.** D13 shares cloning terminology with pilot Q3, but asks which fitted transformer to inspect after cache-triggered cloning. Pilot Q3 asks about interference between estimators sharing mutable RNG state. The remaining pilot intents are manual standardization, random-label feature selection, cross-call grid-search comparability, the `SimpleImputer` default, and a proposed minimum fold count. D21 asks for a different API default. D22 concerns a splitter parameter and has no fold-count requirement.
- **Overlap within this draft.** D10 asks how stratification affects uncertainty inferred from score spread. D15 asks about the goals and limits of grouped stratification. D19 asks for one whole experiment per test fold and a scoring rule. D11 concerns systematic offset, whereas D16 concerns the zero baseline and dataset dependence. D12 asks what to pass into log loss, whereas D23 asks about unshown numerical behavior. D09 concerns score inflation from shuffling temporally related samples. D22 asks what a specific splitter parameter controls. D17 and D21 share the `ColumnTransformer` API but request different behavior. These are 24 distinct proposed families. Keep paired examples in view during independent review.

## Review change log

Milestone 5E clarifies D20’s expected answer, records the reported independent review of D20 and D22, and counts items by primary document. All question wording, family IDs, and proposed evidence spans carry forward.

### Milestone 5D corrections (historical)

The entries below preserve the edit history. The D22 status at the time of the correction is superseded by the independent review outcome above. Repository-owner approval is a separate checkpoint.

| ID and family | Independent review correction |
| --- | --- |
| D05 · `confusion-matrix-axis-convention` | Replaced the same-sentence rationale with the true-row statement and the predicted-column inference from the next paragraph. |
| D09 · `time-series-shuffle-autocorrelation` | Removed the forecasting qualifier and tied score inflation to similarity between time-near training and validation samples. |
| D11 · `explained-variance-systematic-offset` | Replaced the conditional R² preference with the chapter’s general recommendation. |
| D12 · `log-loss-probability-input` | Added the exact `response_method="predict_proba"` sentence from Model Evaluation §3.4.3.2.2 as an alternative for A. |
| D13 · `pipeline-cache-reuse-and-inspection` | Removed both fold-specific claims and the same-fold rationale. Kept identical parameters and input data in A, and clone inspection through `named_steps` in B. |
| D19 · `leave-one-experiment-out-balanced-score` | Removed the sentence about fold data supporting both class recalls. Stated the documented splitter and binary balanced-accuracy identity directly. |
| D20 · `precision-recall-threshold-final-holdout` | Replaced the validation-set sentence with the two preceding test-overfitting and leakage sentences. Added them as B’s second location, labeled the inference, and recorded the alternative document pair. A and B remain the two required units. |
| D21 · `column-transformer-sparse-density-cutoff` | Replaced the question with the scoped request for the default `sparse_threshold` value. |
| D22 · `time-series-split-gap-semantics` | Replaced the test-size formula question with the scoped `gap` question. Rechecked the four chapters and marked the replacement provisional pending human confirmation. |

### Earlier human review edits (historical)

The D20 entry below describes an earlier decision. The later Milestone 5D correction supersedes it and includes Cross-validation as an alternative for B.

| ID and family | Edit after review |
| --- | --- |
| D07 · `random-state-none-global-rng` | Recorded the reviewer’s decision to retain the question and closed the pilot-overlap concern. |
| D13 · `pipeline-cache-reuse-and-inspection` | Specified two grid-search candidates and a fitted pipeline. Added the source’s `memory` and last-step conditions to A, and explained the same-fold reuse example. Kept `named_steps` and the fitted clone in B. |
| D17 · `mixed-text-categorical-fixed-validation` | Added the exact constructor showing `'title'` and `['city']` as an independently sufficient alternative for A. |
| D18 · `log-target-wrapper-and-growth-metric` | Asked explicitly for forward and inverse functions. Replaced the general wrapper excerpt with the contiguous log/exp definitions, constructor and fit. Extended B to include the metric name and formula alongside its original growth recommendation. |
| D19 · `leave-one-experiment-out-balanced-score` | Asked which splitter the Cross-validation chapter uses in its example. Recorded the equivalent configurations for context. |
| D20 · `precision-recall-threshold-final-holdout` | Kept the Model Evaluation and Common Pitfalls excerpts. Made the threshold-to-decision-rule inference explicit. Recorded the Cross-validation passage as corroboration outside the proposed alternatives. |
| D23 · `log-loss-float32-probability-clipping` | Scoped the exact float32 bound request to the four pinned chapters. |
| D24 · `permutation-p-value-monte-carlo-interval` | Scoped the interval recommendation to the four pinned chapters and kept the finite-permutation condition. |

## Corpus validation and remaining semantic risks

The four raw HTML hashes matched the manifest and the processed wrapper references. The existing annotation loader resolved all 31 cited excerpts at the specified sections. In-memory schema-v2 validation passed for all 24 items, including the alternatives for D12, D17, and D20. The allocation remains six direct, six conceptual, four within-document multi-evidence, four cross-document, and four corpus-scoped unanswerable candidates.

The corpus-wide check found the two claims for each cross-document item in distinct documents. The table records the locations for each required unit. The source sets for A and B in each row have no document in common.

| ID | Unit A | Unit B |
| --- | --- | --- |
| D17 | Compose, including the selector-code alternative | Cross-validation |
| D18 | Compose | Model Evaluation |
| D19 | Cross-validation | Model Evaluation |
| D20 | Model Evaluation | Common Pitfalls or Cross-validation |

| Adjacent-field item | Document | Endpoint fields | Canonical range |
| --- | --- | --- | --- |
| D05 | `model_evaluation` | `/blocks/0/text` to `/blocks/1/text` | `[29716, 30075)` |
| D10 | `cross_validation` | `/blocks/4/text` to `/blocks/5/text` | `[18073, 18502)` |

Each span uses adjacent fields in its cited section, with the canonical two-LF separator. D05 pairs the row convention with the true/predicted index definition. D10 pairs the mechanism with its effect on score spread. These coordinates validate the review excerpts. For later retrieval scoring, full-span coverage depends on whether the entire span fits in a retrieved chunk.

- D18’s code demonstrates log/exp configuration. Its inverse-mapping interpretation is stated in the section introduction. Check the A unit as written against that intent and keep the recommendation for MSLE distinct from the training transform.
- D19’s chapter-specific wording resolves which implementation the answer should name. Verify that reviewers consistently read the question that way when they know equivalent splitters.
- D20 requires applying a general holdout rule to a classifier’s decision threshold. The two B passages support the conclusion through that inference, which should remain explicit when presenting the expected answer.
- D21–D24 have chapter-bounded absence findings. For D22, the printed `gap=0` alone cannot establish what the parameter controls. D23 and D24 ask for details that may be known outside the corpus.

## Handoff for owner approval and export

The owner approval record lists the draft revision, scope and sign-off wording. After explicit approval, re-verify all excerpts and coordinates, including the D12 alternative. Preserve D20’s single A location and two locations for B. Each cross-document item requires both units regardless of the alternative selected. Use the family IDs as exported question IDs. Preserve the manifest ID, processed-document revision, section ID, field paths, and document-local Unicode offsets for each approved span. D21–D24 have empty evidence lists. [Owner approval record](development-owner-approval.md)
