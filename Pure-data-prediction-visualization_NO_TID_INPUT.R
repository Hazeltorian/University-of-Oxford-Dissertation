# Libraries

library(dplyr)

# Dealing with output

predictions <- read.csv("pred_best_model_NoTID.csv", header = TRUE)
deltaTpred <- predictions[,1]

pred_test_no_deltaT <- predictions[,-c(1)]

# Function to apply softmax to segments of a row
softmax_segments <- function(row, segment_lengths) {
  start <- 1
  out <- numeric(length(row))
  
  for (seg_len in segment_lengths) {
    end <- start + seg_len - 1
    segment <- row[start:end]
    
    # Numerically stable softmax
    exp_segment <- exp(segment - max(segment))
    out[start:end] <- exp_segment / sum(exp_segment)
    
    start <- end + 1
  }
  
  return(out)
}

# Function to apply softmax_segments to each row of a dataframe
softmax_dataframe <- function(df, segment_lengths) {
  t(apply(df, 1, function(row) softmax_segments(row, segment_lengths)))
}

segment_lengths <- c(9, 12, 14)

# Apply softmax row-wise
pred_test_softmax <- softmax_dataframe(pred_test_no_deltaT, segment_lengths)#

pred_test_softmax_full <- as.data.frame(pred_test_softmax)
pred_test_softmax_full$deltaTpred <- as.numeric(deltaTpred) 
pred_test_softmax_full$original_index <- predictions$original_index

sum(pred_test_softmax_full[1,c(1:9)])  #Should equal 1
sum(pred_test_softmax_full[1,c(10:21)])#Should equal 1
sum(pred_test_softmax_full[1,c(22:35)])#Should equal 1

summary(pred_test_softmax_full[c(22:35)])

#
#
#
#
# Finding which extra rows are missing

library(dplyr)

pred_indices <- predictions$original_index

fmtest_short <- fmtest_short %>%
  mutate(row_id = row_number())

expected_indices <- fmtest_short %>%
  group_by(MID) %>%
  mutate(event_number = row_number()) %>%
  ungroup() %>%
  pull(row_id)

missing_rows <- setdiff(expected_indices, pred_indices)

missing_data <- fmtest_short %>%
  filter(row_id %in% missing_rows) %>%
  arrange(MID, row_id)

pred_F1 <- fmtest_short[-c(missing_rows),]


#
#
#
#
# Coding up prediction visualizations

# Vector of team IDs in the order of prediction columns
team_ids <- c(971, 746, 970, 968, 967, 969, 974, 973, 965, 966, 972, 749, 1475, 2647)
# Map team IDs to actual columns (22–35)
tid_to_col <- setNames(22:35, team_ids)

# Correctly extract the probabilities
pred_F1$prob_correct <- pred_test_softmax_full[
  cbind(seq_len(nrow(pred_test_softmax_full)),
        tid_to_col[as.character(pred_F1$TID)])
]
# Add event index within each match
pred_F1 <- pred_F1 %>%
  group_by(MID) %>%
  mutate(event_number = row_number()) %>%
  ungroup()

#
#
#
#
#

team_ids <- sort(unique(fmtraining_short$TID))


# 1. Extract highest predicted team per event
predicted_indices <- apply(pred_test_softmax_full[,22:35], 1, which.max)
predicted_teams <- team_ids[predicted_indices]

# 2. Extract true team for each event
# Assuming pred_F1$TID contains the true team for each event in the same order
true_teams <- pred_F1$TID

# 3. Create confusion matrix
confusion_mat <- matrix(0, nrow = length(team_ids), ncol = length(team_ids),
                        dimnames = list(predicted = team_ids, true = team_ids))

# 4. Fill the matrix
for(i in seq_along(predicted_teams)) {
  pred <- as.character(predicted_teams[i])
  true <- as.character(true_teams[i])
  confusion_mat[pred, true] <- confusion_mat[pred, true] + 1
}

# 5. Convert to data frame for easier viewing (optional)
confusion_df_NOTID <- as.data.frame(confusion_mat)

# 6. Quick check: sum of all entries should equal total number of events
sum(confusion_df_NOTID)

#7 sum each row and divide the diagonal digit by the total

diag(confusion_df_NOTID)/colSums(confusion_df_NOTID)

