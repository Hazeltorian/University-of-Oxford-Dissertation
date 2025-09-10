# Libraries

library(dplyr)

# Dealing with output

predictions <- read.csv("pred_best_model_TID_included.csv", header = TRUE)
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

#
#
#
#
# Coding up prediction visualizations

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

#
#
#
#
#

# Vector of team IDs corresponding to columns 22–35
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
confusion_df_TIDFA <- as.data.frame(confusion_mat)
confusion_mat_TIDFA <- as.matrix(confusion_df_TIDFA)

# 6. Quick check: sum of all entries should equal total number of events
sum(confusion_df_TIDFA)

#7 sum each row and divide the diagonal digit by the total

diag(confusion_mat_TIDFA)/colSums(confusion_mat_TIDFA)

diag(confusion_mat_TIDFA)/rowSums(confusion_mat_TIDFA)


#
#
#
#
#

fm.19_home_teams_id <- fm.19$home_team.home_team_id
fm.20_home_teams_id <- fm.20$home_team.home_team_id
fm.21_home_teams_id <- fm.21$home_team.home_team_id

fm.19_away_teams_id <- fm.19$away_team.away_team_id
fm.20_away_teams_id <- fm.20$away_team.away_team_id
fm.21_away_teams_id <- fm.21$away_team.away_team_id

all_ids_19 <- c(fm.19_home_teams_id, fm.19_away_teams_id)
counts_19 <- table(all_ids_19)

all_ids_20 <- c(fm.20_home_teams_id, fm.20_away_teams_id)
counts_20 <- table(c(fm.20_home_teams_id, fm.20_away_teams_id))

all_ids_21 <- c(fm.21_home_teams_id, fm.21_away_teams_id)
counts_21 <- table(c(fm.21_home_teams_id, fm.21_away_teams_id))

all_ids <- sort(unique(c(names(counts_19), names(counts_20), names(counts_21))))

vec_19 <- setNames(as.numeric(counts_19[all_ids]), all_ids)
vec_19[is.na(vec_19)] <- 0

vec_20 <- setNames(as.numeric(counts_20[all_ids]), all_ids)
vec_20[is.na(vec_20)] <- 0

vec_21 <- setNames(as.numeric(counts_21[all_ids]), all_ids)
vec_21[is.na(vec_21)] <- 0

grand_total <- vec_19 + vec_20 + vec_21
grand_total

fm.19_home_teams_name <- unique(fm.19$home_team.home_team_name) 
fm.20_home_teams_name <- unique(fm.20$home_team.home_team_name) 
fm.21_home_teams_name <- unique(fm.21$home_team.home_team_name) 

all_data <- rbind(fmtraining_short, fmvalid_short, fmtest_short[,-7])
team_event_counts <- table(all_data$TID)
team_event_counts_df <- as.data.frame(team_event_counts)

team_event_counts_df$proportion <- team_event_counts_df$Freq / grand_total[as.character(team_event_counts_df$Var1)]

# Number of events per game

team_event_counts_df <- team_event_counts_df[order(-team_event_counts_df$proportion), ]


#
#
#
#
#

# Determine match results
fm.19$result <- with(fm.19, ifelse(home_score > away_score, "home_win",
                            ifelse(home_score < away_score, "away_win", "draw")))

# Create a column for winner_id (NA for draws)
fm.19$winner_id <- with(fm.19, ifelse(result == "home_win", home_team.home_team_id,
                               ifelse(result == "away_win", away_team.away_team_id, NA)))

# For draws, store BOTH teams in a separate column as a list
fm.19$draw_teams <- with(fm.19, ifelse(result == "draw", 
                                        paste(home_team.home_team_id, away_team.away_team_id, sep = ","), 
                                        NA))


# --- fm.20 ---
fm.20$result <- with(fm.20, ifelse(home_score > away_score, "home_win",
                            ifelse(home_score < away_score, "away_win", "draw")))

fm.20$winner_id <- with(fm.20, ifelse(result == "home_win", home_team.home_team_id,
                               ifelse(result == "away_win", away_team.away_team_id, NA)))

fm.20$draw_teams <- with(fm.20, ifelse(result == "draw", 
                                        paste(home_team.home_team_id, away_team.away_team_id, sep = ","), 
                                        NA))

# --- fm.21 ---
fm.21$result <- with(fm.21, ifelse(home_score > away_score, "home_win",
                            ifelse(home_score < away_score, "away_win", "draw")))

fm.21$winner_id <- with(fm.21, ifelse(result == "home_win", home_team.home_team_id,
                               ifelse(result == "away_win", away_team.away_team_id, NA)))

fm.21$draw_teams <- with(fm.21, ifelse(result == "draw", 
                                        paste(home_team.home_team_id, away_team.away_team_id, sep = ","), 
                                        NA))

league_19 <- calculate_league_table(fm.19)

# League table for 2020
league_20 <- calculate_league_table(fm.20)

# League table for 2021
league_21 <- calculate_league_table(fm.21)
#
#
#
#
#

# Ppg per season

league_19$ppg <- league_19$points/league_19$played
league_20$ppg <- league_20$points/league_20$played
league_21$ppg <- league_21$points/league_21$played

# std ppg

league_19$season <- 2019
league_20$season <- 2020
league_21$season <- 2021

all_leagues <- rbind(league_19, league_20, league_21)

team_ppg_sd <- all_leagues %>%
  group_by(team_id) %>%
  summarise(
    sd_ppg = sd(ppg),
    mean_ppg = mean(ppg),
    seasons_played = n()
  ) %>%
  arrange(desc(sd_ppg))

Sensitivity <- diag(confusion_mat_TIDFA)/colSums(confusion_mat_TIDFA)

Specificity <- 1-((rowSums(confusion_mat_TIDFA) - diag(confusion_mat_TIDFA)) / (sum(confusion_mat_TIDFA)-colSums(confusion_mat_TIDFA)))

sensitivity <- c(
  `746` = 0.9685070, `749` = 0.9856233, `965` = 0.9306477, `966` = 0.9992745,
  `967` = 0.8488613, `968` = 0.9758212, `969` = 0.9681794, `970` = 1.0000000,
  `971` = 0.9823821, `972` = 0.9089270, `973` = 0.9017055, `974` = 0.9353951,
  `1475` = 0.9964935, `2647` = 0.9971559
)

specificity <- c(
  `746`  = 0.9985044,`749`  = 0.9982299,`965`  = 0.9947881,`966`  = 0.9913825,
  `967`  = 0.9997504,`968`  = 0.9995219,`969`  = 0.9979723,`970`  = 0.9851460,
  `971`  = 0.9936022,`972`  = 0.9991580,`973`  = 0.9978610,`974`  = 0.9998844,
  `1475` = 0.9995093,`2647` = 0.9972782
)

# Add to team_ppg_sd tibble
team_ppg_sd <- team_ppg_sd %>%
  dplyr::mutate(
    sensitivity = sensitivity[as.character(team_id)],
    specificity = specificity[as.character(team_id)]
  )

team_ppg_sd <- as.data.frame(team_ppg_sd) 

#
#
#
#
#

#looking for links

team_accuracy <- diag(confusion_mat_TIDFA) / colSums(confusion_mat_TIDFA)
team_spec <- 1-((rowSums(confusion_mat_TIDFA) - diag(confusion_mat_TIDFA)) / (sum(confusion_mat_TIDFA)-colSums(confusion_mat_TIDFA)))

# Convert to a data frame
team_accuracy_df <- data.frame(
  team_id = as.integer(names(team_accuracy)),
  accuracy = as.numeric(team_accuracy)
)

team_spec_df <- data.frame(
  team_id = as.integer(names(team_spec)),
  accuracy = as.numeric(team_spec )
)

team_ppg_sd <- team_ppg_sd %>%
  left_join(team_accuracy_df, by = "team_id")

team_ppg_sd <- team_ppg_sd %>%
  left_join(team_spec_df, by = "team_id")

fit_spec <- lm(sensitivity ~ mean_ppg + I(mean_ppg^2), data = team_ppg_sd)

# Create a sequence of x-values for a smooth curve
x_vals <- seq(min(team_ppg_sd$mean_ppg), max(team_ppg_sd$mean_ppg), length.out = 200)

# Predict y-values from the quadratic model
y_vals <- predict(fit_spec, newdata = data.frame(mean_ppg = x_vals))

# Add the curve
lines(x_vals, y_vals, col = "red", lwd = 2)

pdf("Senplot.pdf", width = 8, height = 5)

plot(team_ppg_sd$sensitivity ~ team_ppg_sd$mean_ppg,
xlab = "Mean Points Per Game",
ylab = "Sensitivity")
lines(x_vals, y_vals, col = "red", lwd = 2)

dev.off()


pdf("Ppvplot.pdf", width = 8, height = 5)
scatter.smooth(team_ppg_sd$spec ~ team_ppg_sd$mean_ppg,
xlab = "Mean Points Per Game",
ylab = "Positive Predictive Value",lpars =
                    list(col = "red", lwd = 2, lty = 1))
dev.off()


pdf("specplot.pdf", width = 8, height = 5)
scatter.smooth(team_ppg_sd$spec ~ team_ppg_sd$mean_ppg,
xlab = "Mean Points Per Game",
ylab = "Specificity",lpars =
                    list(col = "red", lwd = 2, lty = 1))
dev.off()



