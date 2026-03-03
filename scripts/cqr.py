import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class ConformalQuantileRegressor:
    """
    Conformalized Quantile Regression: constructs prediction intervals with 
    target coverage (1 - alpha), combining quantile regression and conformal calibration.
    """
    def __init__(
        self,
        alpha=0.1,
        quantile_regressor=None,
        random_state=None,
        reg_alpha=None,
        **model_kwargs,
    ):
        """
        Initialize with desired miscoverage alpha (e.g. alpha=0.1 for 90% coverage).
        Optional: provide a sklearn regressor class for quantile modeling (default uses GradientBoostingRegressor).
        Any additional model_kwargs are passed to the regressor (e.g. n_estimators, max_depth).

        Supported quantile regressor APIs (chosen automatically by trying constructor kwargs):
          1) Tree-style API (e.g. sklearn.ensemble.GradientBoostingRegressor):
               model_class(loss="quantile", alpha=tau, **kwargs)
             where tau is the quantile level in (0,1).
          2) Linear QuantileRegressor API (sklearn.linear_model.QuantileRegressor):
               model_class(quantile=tau, alpha=reg_alpha, **kwargs)
             Note: QuantileRegressor's `alpha` is *regularization strength*, not the quantile.
        """
        self.alpha = alpha
        self.alpha_lo = alpha/2
        self.alpha_hi = 1 - alpha/2
        # Use provided regressor class or default to GradientBoostingRegressor
        self.model_class = quantile_regressor or GradientBoostingRegressor
        self.random_state = random_state
        self.reg_alpha = reg_alpha
        self.model_kwargs = model_kwargs
        self.lower_model = None
        self.upper_model = None
        self.calibration_offset = 0.0

        # Disallow passing a *fitted instance*; we need to instantiate two fresh models with different taus.
        if self.model_class is not None and not isinstance(self.model_class, type):
            raise TypeError(
                "quantile_regressor must be a regressor CLASS (e.g. GradientBoostingRegressor or QuantileRegressor), "
                f"got instance of type {type(self.model_class)}"
            )

    def _make_model(self, tau: float):
        """
        Instantiate a quantile model targeting quantile level tau in (0,1).
        Tries multiple common sklearn constructor conventions.
        """
        if not (0.0 < float(tau) < 1.0):
            raise ValueError(f"tau must be in (0,1), got {tau}")

        base = dict(self.model_kwargs)
        # QuantileRegressor has an `alpha` regularization hyperparameter; keep it distinct
        # from CQR miscoverage `alpha`. Users should pass it as `reg_alpha=`.
        if self.reg_alpha is not None and "alpha" not in base:
            base["alpha"] = self.reg_alpha
        # Pass random_state only if provided; if the model doesn't accept it we will retry without.
        if self.random_state is not None and "random_state" not in base:
            base["random_state"] = self.random_state

        # Candidate constructor kwargs in order of preference.
        candidates = []

        # (1) GradientBoostingRegressor-style quantile loss: loss="quantile", alpha=tau.
        kw1 = dict(base)
        kw1.pop("loss", None)
        kw1["loss"] = "quantile"
        kw1["alpha"] = float(tau)  # for GBR this is the quantile level
        candidates.append(kw1)

        # (2) sklearn.linear_model.QuantileRegressor-style: quantile=tau.
        kw2 = dict(base)
        kw2.pop("loss", None)
        kw2["quantile"] = float(tau)
        candidates.append(kw2)

        last_err = None
        for kw in candidates:
            try:
                return self.model_class(**kw)
            except TypeError as e:
                last_err = e
                # Common: model doesn't accept random_state. Retry once without it.
                if "random_state" in kw:
                    kw2b = dict(kw)
                    kw2b.pop("random_state", None)
                    try:
                        return self.model_class(**kw2b)
                    except TypeError as e2:
                        last_err = e2
                        continue
                continue

        raise TypeError(
            "Could not instantiate quantile regressor with a supported API. "
            "Tried loss='quantile', alpha=tau and quantile=tau. "
            f"Last error: {last_err}"
        )

    def fit(self, X_train, y_train, X_calib, y_calib):
        """
        Train quantile regression models on (X_train, y_train) and calibrate on (X_calib, y_calib).
        """
        # 1. Train lower quantile model (tau = alpha/2)
        self.lower_model = self._make_model(self.alpha_lo)
        self.lower_model.fit(X_train, y_train)
        # 2. Train upper quantile model (tau = 1 - alpha/2)
        self.upper_model = self._make_model(self.alpha_hi)
        self.upper_model.fit(X_train, y_train)
        # 3. Apply models to calibration set
        q_lo_pred = self.lower_model.predict(X_calib)
        q_hi_pred = self.upper_model.predict(X_calib)
        # 4. Compute nonconformity scores on calibration data
        #    Score = max(0, q_lo_pred - y, y - q_hi_pred)
        diff_lo = q_lo_pred - y_calib         # positive if y is below the lower bound
        diff_hi = y_calib - q_hi_pred         # positive if y is above the upper bound
        scores = np.maximum(0, np.maximum(diff_lo, diff_hi))
        # 5. Determine the 1-alpha quantile of scores for calibration (with conservative rounding up)
        sorted_scores = np.sort(scores)
        m = len(sorted_scores)
        cutoff_index = int(np.ceil((1 - self.alpha) * (m + 1))) - 1  # quantile index (0-based)
        cutoff_index = min(max(cutoff_index, 0), m - 1)
        self.calibration_offset = sorted_scores[cutoff_index]
        return self  # enables method chaining

    def predict_interval(self, X):
        """
        Given new input features X (array of shape [n_samples, n_features]), 
        return the prediction interval [lower, upper] for each sample.
        """
        if self.lower_model is None or self.upper_model is None:
            raise RuntimeError("Call fit() first to train and calibrate the model.")
        # Predict quantiles
        q_lo = self.lower_model.predict(X)
        q_hi = self.upper_model.predict(X)
        # Apply the calibrated offset
        lower_bound = q_lo - self.calibration_offset
        upper_bound = q_hi + self.calibration_offset
        # Ensure no negative interval width (due to any model inaccuracies)
        # If lower_bound > upper_bound, adjust to make it a point interval
        mask = lower_bound > upper_bound
        if np.any(mask):
            lower_bound[mask] = upper_bound[mask]
        return lower_bound, upper_bound
