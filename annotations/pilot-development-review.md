# DocGround development pilot review

**Human review approved.** The development dataset is [pilot-v1.json](development/pilot-v1.json). The reviewer approved all Q1–Q8 evidence mappings, kept both units for Q1, Q4, Q6, and Q8, and approved Q9 and Q10 as unanswerable within this chapter. Q9 uses the approved revised wording.

## Pinned source and verification

The corpus is chapter 12, *Common pitfalls and recommended practices*, from the local scikit-learn 1.9.1 snapshot. The source of record is `data/raw/common_pitfalls.html` and its canonical representation is `data/processed/common_pitfalls.json`. Section anchors below identify locations in that snapshot.

- Dataset version `pilot-v1`, annotation schema `1`, split `development`
- Exported UTF-8/LF JSON SHA-256 `8b805eff016697513f8e2380db3ca52d7d1d3b58cb3ce13208bc79b1b9cfac16`
- Raw HTML SHA-256 `21fac11c181ab35624dcf8673f0475d403d4f6ecf0f9bead01a573d7642b515c`
- Processed JSON file SHA-256 `460654b1b7277ba589dd5856eecd1d871074889f4067f9d9766944ee6c8f27f5`
- Canonical `corpus_revision` `7c96bf1e360e8d3561b98313409323d2c074371cd71fa5599d50c247f03f652e`

The provisional review checked all 15 processed sections and the raw HTML article for Q9 and Q10. The final export was validated with the annotation CLI against the pinned canonical document. All 12 selected spans and both context excerpts match the exact canonical text at their documented coordinates. The dataset has ten questions, eight answerable and two unanswerable. The frozen-test directory contains only its original placeholder.

Coordinates below count zero-based Unicode code points in the canonical document and use half-open ranges `[start, end)`. A single block path serves as both endpoint paths. Full section IDs appear in the locator table.

## Approved development questions

| Question | Answerability | Expected answer | Evidence |
| --- | --- | --- | --- |
| **Q1.** A regression model was trained on standardized features. What should happen to `X_test` before calling `predict` in the manual workflow? | Answerable, two units | Apply the scaler already fitted on training data, then pass the transformed test data to the model. | [Q1.A](#q1-a), [Q1.B](#q1-b) |
| **Q2.** Why does the chapter’s random-label feature-selection example score well above chance when feature selection runs before the train/test split? | Answerable | The features and labels are independent, so the example should be near chance. Selecting features from the full dataset lets information from the eventual test set influence which features are retained. | [Q2.A](#q2-a) |
| **Q3.** In the chapter’s cloning example, why can fitting one random forest affect the other after an explicit `RandomState` instance is passed? | Answerable | The two estimators share the same RNG. Fitting one consumes its state, which changes the state available to the other. | [Q3.A](#q3-a) |
| **Q4.** Where should supervised feature selection go when you use `cross_val_score` to evaluate a classifier, and why? | Answerable, two units | Put the selector and classifier in one pipeline and pass that pipeline to `cross_val_score`. The selector is fitted on each fold’s training subset, and its learned transformation is applied to the held-out subset. | [Q4.A](#q4-a), [Q4.B](#q4-b) |
| **Q5.** Why can calculating a preprocessing statistic, such as the mean, using the full dataset before a train/test split cause data leakage, even if the same transformation is applied consistently afterward? | Answerable | A statistic calculated on all the data incorporates information from the eventual test subset into preprocessing that affects the model. Estimate it using the training subset and reuse it for both subsets. | [Q5.A](#q5-a) |
| **Q6.** In repeated cross-validation comparisons, what does an integer `random_state` control on a randomized estimator, and what does it control on a shuffled CV splitter? | Answerable, two units | On the estimator, each fit starts from the same RNG state. On the splitter, repeated split calls yield the same folds, making fold-by-fold comparisons possible. | [Q6.A](#q6-a), [Q6.B](#q6-b) |
| **Q7.** With a `RandomState` instance on a shuffled splitter, why can candidates within one `GridSearchCV.fit` call be compared fold by fold while scores from two separate calls should be treated differently? | Answerable | Within one search call, the splitter is invoked once and the candidates use the same folds. Separate calls can invoke the splitter again and use different folds. | [Q7.A](#q7-a) |
| **Q8.** How can you obtain reproducible results across separate program executions while allowing a randomized estimator to use different random states across cross-validation folds? | Answerable, two units (synthesis) | Replace `random_state=None` with explicitly seeded randomness. Pass a freshly created, seeded `RandomState` instance to the estimator at the start of each run so the sequence can repeat across executions and vary across fold fits. | [Q8.A](#q8-a), [Q8.B](#q8-b) |
| **Q9.** According to the pinned Common Pitfalls chapter, what is the default value of the `strategy` parameter in `SimpleImputer`? | Unanswerable in this chapter | The pinned chapter mentions `SimpleImputer` as a transformation with leakage risk and gives no default value for `strategy`. | [Q9 context](#q9-context) |
| **Q10.** What minimum number of CV folds does the chapter prescribe before average scores can be used to compare two estimators evaluated on different splits? | Unanswerable in this chapter | The chapter says average scores may be informative with enough folds and data, and supplies no numeric minimum. | [Q10 context](#q10-context) |

## Exact excerpts and coordinates

Every approved unit below has one span. Alternative passages require independent support for the same claim. Q9 and Q10 have empty evidence arrays. Their context is shown for inspection only. Multiline excerpts use JSON string notation so each `\n` denotes one actual LF character.

### Q1 A

**Location** / `#inconsistent-preprocessing` / `/blocks/5/text` / `[1434, 1554)`

> `">>> X_train_transformed = scaler.fit_transform(X_train)\n>>> model = LinearRegression().fit(X_train_transformed, y_train)"`

Establishes that the scaler was fitted on training data and that the model was trained on its output.

### Q1 B

**Location** / `#inconsistent-preprocessing` / `/blocks/8/text` / `[1768, 1883)`

> `">>> X_test_transformed = scaler.transform(X_test)\n>>> mean_squared_error(y_test, model.predict(X_test_transformed))"`

Shows that the same `scaler` transforms `X_test` and that the model predicts on the transformed values.

### Q2 A

**Location** / `#data-leakage-during-pre-processing` / `/blocks/6/text` / `[5397, 5749)`

> Using all the data to perform feature selection results in an accuracy score much higher than chance, even though our targets are completely random. This randomness means that our X and y are independent and we thus expect the accuracy to be around 0.5. However, since the feature selection step ‘sees’ the test data, the model has an unfair advantage.

States the chance-level expectation and identifies test-set exposure during feature selection as the reason the score is inflated.

### Q3 A

**Location** / `#id2` / `/blocks/8/blocks/2/text` / `[15914, 16098)`

> Moreover, a and b will influence each other since they share the same internal RNG: calling a.fit will consume b’s RNG, and calling b.fit will consume a’s RNG, since they are the same.

The Cloning panel explicitly identifies the shared RNG and describes how either fit consumes it. `a` and `b` are defined in the preceding code.

### Q4 A

**Location** / `#data-leakage-during-pre-processing` / `/blocks/11/text` / `[7390, 7637)`

> Here again, we recommend using a Pipeline to chain together the feature selection and model estimators. The pipeline ensures that only the training data is used when performing fit and the test data is used only for calculating the accuracy score:

Establishes the selector’s placement in the pipeline and the training-only fit rule.

### Q4 B

**Location** / `#data-leakage-during-pre-processing` / `/blocks/13/text` / `[8184, 8387)`

> The pipeline can also be fed into a cross-validation function such as cross_val_score. Again, the pipeline ensures that the correct data subset and estimator method is used during fitting and predicting:

Applies that pipeline to CV and states why it routes fitting and prediction correctly. The fold-wise transform follows from these two excerpts together with the held-out-data behavior illustrated by the adjacent example.

### Q5 A

**Location** / `#data-leakage` / `/blocks/2/text` / `[3117, 3625)`

> Although both train and test data subsets should receive the same preprocessing transformation (as described in the previous section), it is important that these transformations are only learnt from the training data. For example, if you have a normalization step where you divide by the average value, the average should be the average of the train subset, not the average of all the data. If the test subset is included in the average calculation, information from the test subset is influencing the model.

Explicitly distinguishes applying the same transform from learning it on the correct subset and gives the mean example.

### Q6 A

**Location** / `#id2` / `/blocks/4/items/0/0/text` / `[14224, 14432)`

> Since rf_123 was passed an integer, every call to fit uses the same RNG: this means that all random characteristics of the random forest estimator will be the same for each of the 5 folds of the CV procedure.

Supports the estimator fit behavior across folds in the random forest example.

### Q6 B

**Location** / `#using-none-or-randomstate-instances-and-repeated-calls-to-fit-and-split` / `/blocks/1/items/0/0/text` / `[9823, 9915)`

> If an integer is passed, calling fit or split multiple times always yields the same results.

Explicitly covers repeated `split` calls. The comparison-specific application is illustrated in section 12.3.2.2, `/blocks/3/text`.

### Q7 A

**Location** / `#id3` / `/blocks/2/text` / `[17768, 18157)`

> This is also true for any tool that performs model selection via cross-validation, e.g. GridSearchCV and RandomizedSearchCV: scores are not comparable fold-to-fold across different calls to search.fit, since cv.split would have been called multiple times. Within a single call to search.fit, however, fold-to-fold comparison is possible since the search estimator only calls cv.split once.

Directly contrasts the number of split calls and the comparability of fold-level scores within and across search fits.

### Q8 A

**Location** / `#getting-reproducible-results-across-multiple-executions` / `/blocks/0/text` to `/blocks/1/text` / `[18805, 19438)`

> `"In order to obtain reproducible (i.e. constant) results across multiple program executions, we need to remove all uses of random_state=None, which is the default. The recommended way is to declare a rng variable at the top of the program, and pass it down to any object that accepts a random_state parameter:\n\n>>> from sklearn.ensemble import RandomForestClassifier\n>>> from sklearn.datasets import make_classification\n>>> from sklearn.model_selection import train_test_split\n>>> import numpy as np\n\n>>> rng = np.random.RandomState(0)\n>>> X, y = make_classification(random_state=rng)\n>>> rf = RandomForestClassifier(random_state=rng)"`

Combines the across-execution recommendation with its concrete initialization and assignment to the estimator.

### Q8 B

**Location** / `#robustness-of-cross-validation-results` / `/blocks/1/text` / `[20746, 20975)`

> For these reasons, it is preferable to evaluate the cross-validation performance by letting the estimator use a different RNG on each fold. This is done by passing a RandomState instance (or None) to the estimator initialization.

Establishes how to vary estimator randomness by fold. Unit A is why the combined answer chooses a seeded instance instead of `None`.

### Q9 context

**Location** / `#data-leakage-during-pre-processing` / `/blocks/1/text` / `[4545, 4705)`

> This risk of leakage is however relevant with almost all transformations in scikit-learn, including (but not limited to) StandardScaler, SimpleImputer, and PCA.

The sole mention in the pinned chapter names the class without specifying parameters. This is inspection context only.

### Q10 context

**Location** / `#id3` / `/blocks/5/text` / `[18309, 18533)`

> While fold-to-fold comparison is not advisable with RandomState instances, one can however expect that average scores allow to conclude whether one estimator is better than another, as long as enough folds and data are used.

This is a qualitative condition. The two-fold and five-fold examples elsewhere illustrate different points and give no rule about a minimum.

## Section locators

| Snapshot section | Anchor | Full section ID |
| --- | --- | --- |
| 12.1. Inconsistent preprocessing | `#inconsistent-preprocessing` | `6025f191b831f2a0e2b98b20628af96f6bba3223c90a535a94967776413e5208` |
| 12.2.2. Data leakage during pre-processing | `#data-leakage-during-pre-processing` | `79e3ef9172f09b14f07b25d1bf1d25d2389a8a7f2516b435fc297728efba49f2` |
| 12.3.2.1. Estimators | `#id2` | `7fcfa8b68c9213e5a1995b55a955e852f03636fcb37282f5ffd0751b82b2d849` |
| 12.2. Data leakage | `#data-leakage` | `06b68e3475a3af161c1ee8f39c05b077063b42e0bb3fc0ef1d582017f83484d3` |
| 12.3.1. Using None or RandomState instances, and repeated calls to fit and split | `#using-none-or-randomstate-instances-and-repeated-calls-to-fit-and-split` | `f0981bb86b47205556874123f5ab1486b018dde54422c25f036e008fe34a74e1` |
| 12.3.2.2. CV splitters | `#id3` | `169529d4a1df856e180c9ded0aaaed29644f0a6e920cc18fb7ca559696ec0fe3` |
| 12.3.3.1. Getting reproducible results across multiple executions | `#getting-reproducible-results-across-multiple-executions` | `837b41ce7ca7c97d9f2b8be6911423d49a7645099d7a3d2536f67987fa361f7e` |
| 12.3.3.2. Robustness of cross-validation results | `#robustness-of-cross-validation-results` | `da463f75dc63c144634de8e52cb49218df9974a0ca1754b35153ac4da33062f6` |

## Review notes

- **Q1** uses both code spans. One shows that the scaler is fitted on training data, and the other shows that this scaler transforms the test data before prediction.
- **Q4** uses both passages. Together they support putting the selector inside the pipeline and inferring the fold-wise transformation from the training-only fit rule and the cross-validation guidance.
- **Q6** uses both passages. The general repeated-call rule supplies the splitter behavior, and the separate shuffled `KFold` example is useful context for the comparison.
- **Q8** assumes that each program execution creates a fresh `np.random.RandomState` with the same fixed seed and passes the instance to the estimator. A fixed execution environment, inputs, and call order allow the sequence to repeat across runs while RNG state can progress across fold fits within a run. The two approved units support this synthesis from separate examples. The chapter does not demonstrate both properties in a single script.
- **Q9 and Q10** are unanswerable within the pinned chapter. The chapter mentions `SimpleImputer` without specifying the default `strategy` and gives qualitative guidance about the number of folds without a numeric minimum.
