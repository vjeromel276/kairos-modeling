
CREATE TABLE feat_composite_v4 AS
SELECT DISTINCT
    ticker,
    date,
    alpha_composite_eq,
    alpha_CL,
    (0.5) * alpha_composite_eq
  + (0.5) * alpha_CL AS alpha_composite_v4
FROM feat_matrix
WHERE alpha_composite_eq IS NOT NULL
  AND alpha_CL IS NOT NULL;
