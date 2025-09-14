# Libraries

library(dplyr)
library(pheatmap)
library(viridis)

# Dealing with output

predictions <- read.csv("pred_best_model_20_win.csv", header = TRUE)
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
# Code to tell me which Teams play in each match

match_teams_sep <- match_teams %>%
  mutate(
    team1 = sapply(teams_in_match, `[`, 1),
    team2 = sapply(teams_in_match, `[`, 2)
  ) %>%
  select(MID, team1, team2)

match_teams_sep <- as.data.frame(match_teams_sep)

#
#
#
#
#

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

# Vector of team IDs in the order of prediction columns
sort(unique(fmtraining_short$TID))

team_ids <- sort(unique(fmtraining_short$TID))

# Map team IDs to actual columns (22–35)
tid_to_col <- setNames(22:35, team_ids)

pred_F1 <- fmtest_short[-c(missing_rows),]

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
confusion_df_NotestTID <- as.data.frame(confusion_mat)

# 6. Quick check: sum of all entries should equal total number of events
sum(confusion_df_NotestTID)

#7 sum each row and divide the diagonal digit by the total

confusion_mat_NotestTID <- as.matrix(confusion_df_NotestTID)

diag(confusion_mat_NotestTID)/colSums(confusion_mat_NotestTID)

no_test_TID_Heat_map_CM <- apply(confusion_mat_NotestTID, 2, function(x) x / sum(x))

pdf("TID_no_test_heatplot.pdf", width = 8, height = 5)
pheatmap(no_test_TID_Heat_map_CM, scale = "none",
  color = cividis(100),   # or magma(), plasma(), inferno(), cividis()
  legend = TRUE,
  cluster_rows = FALSE,
  cluster_cols = FALSE)
dev.off()



