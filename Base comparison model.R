# logistic regression

# Loop that goes through every match

# Get all unique TIDs in ascending order
all_tids <- sort(unique(fmtest_short$TID))

# Initialize an empty matrix to store counts
agg_conf_matrix <- matrix(0, nrow = length(all_tids), ncol = length(all_tids),
                          dimnames = list(Predicted = all_tids, Actual = all_tids))

# Loop over each MID
for(mid in unique(fmtest_short$MID)) {
  
  match_data <- fmtest_short[fmtest_short$MID == mid, ]
  
  # Skip if not enough rows
  if(nrow(match_data) < 2) next
  
  # Prepare the GLM dataset
  match_df <- data.frame(
    Pred_var = as.factor(match_data$TID[-1]),
    Res_TID = as.factor(match_data$TID[-nrow(match_data)]),
    Res_Act = as.factor(match_data$act[-nrow(match_data)]),
    Res_deltaT = match_data$deltaT[-nrow(match_data)],
    Res_zone = as.factor(match_data$zone[-nrow(match_data)])
  )
  
  # Fit GLM
  match_glm <- glm(Pred_var ~ ., family = binomial, data = match_df)
  
  # Predict
  match_pred <- ifelse(
    predict(match_glm, type = "response") < 0.5,
    levels(match_df$Pred_var)[1],
    levels(match_df$Pred_var)[2]
  )
  match_pred <- as.factor(match_pred)
  
  # Compute confusion matrix for this match
  conf_matrix <- table(match_pred, match_df$Pred_var)
  
  # Add to aggregated matrix
  # Use rownames and colnames to map correctly
  rows <- rownames(conf_matrix)
  cols <- colnames(conf_matrix)
  agg_conf_matrix[rows, cols] <- agg_conf_matrix[rows, cols] + conf_matrix
}

# Convert to data frame if needed for display
agg_conf_df <- as.data.frame.matrix(agg_conf_matrix)
agg_conf_df

diag(as.matrix(agg_conf_df))/colSums(agg_conf_df)
