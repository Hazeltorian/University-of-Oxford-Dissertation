# Package installation

install.packages("devtools")
devtools::install_github("cran/SDMTools") 
devtools::install_github("statsbomb/StatsBombR")
library(StatsBombR)
library(dplyr)

# Read in data from Women's Super League

comp <- FreeCompetitions() 

WSL19 <- comp[which(comp$competition_id == 37 & comp$season_id == 4),] # Games from Women's Super League 2018/2019 season
WSL20 <- comp[which(comp$competition_id == 37 & comp$season_id == 42),]
WSL21 <- comp[which(comp$competition_id == 37 & comp$season_id == 90),]

# Indexes matches from Women's Super League

fm.19 <- FreeMatches(WSL19) 
fm.20 <- FreeMatches(WSL20)
fm.21 <- FreeMatches(WSL21)

# Read events into a tibble format

events.df19 <- events.df20 <- events.df21 <- tibble()

for (i in 1:length(fm.19$match_id)) {
  events <- get.matchFree(fm.19[i, ])
  events.df19 <- bind_rows(events.df19, events)  
}

for (i in 1:length(fm.20$match_id)) {
  events <- get.matchFree(fm.20[i, ])
  events.df20 <- bind_rows(events.df20, events)  
}

for (i in 1:length(fm.21$match_id)) {
  events <- get.matchFree(fm.21[i, ])
  events.df21 <- bind_rows(events.df21, events)   
}


fmd19 <- events.df19

fmd20 <- events.df20

fmd21 <- events.df21

#
#
#
#
#

# Checking what columns are not shared in the different data frames

colsfmd19 <- colnames(fmd19)
colsfmd20 <- colnames(fmd20)
colsfmd21 <- colnames(fmd21)

shared_all <- Reduce(intersect, list(colsfmd19, colsfmd20, colsfmd21))
all_cols <- Reduce(union, list(colsfmd19, colsfmd20, colsfmd21))
not_shared <- setdiff(all_cols, shared_all)

# Making sure all columns are seen in all the different datasets

fmd19 <- fmd19 [ , setdiff(colnames(fmd19 ), not_shared), drop = FALSE]
fmd20 <- fmd20 [ , setdiff(colnames(fmd20 ), not_shared), drop = FALSE]
fmd21 <- fmd21 [ , setdiff(colnames(fmd21 ), not_shared), drop = FALSE]

#
#
#
#
#

# Data set up for Transformer

# Setting up zones

fmd19$Dis_location <- fmd19$location

# Extract x (first coordinate), treating NULLs as NA
x_loc <- unlist(lapply(fmd19$location, function(coord) {
  if (is.null(coord) || length(coord) < 1) NA else coord[1]
}))

# Extract y (second coordinate), treating NULLs as NA
y_loc <- unlist(lapply(fmd19$location, function(coord) {
  if (is.null(coord) || length(coord) < 2) NA else coord[2]
}))

fmd19$Dis_location[x_loc <= 40 & y_loc <= 26.67] <- 1
fmd19$Dis_location[x_loc <= 40 & y_loc > 26.67 & y_loc <= 53.33] <- 2
fmd19$Dis_location[x_loc <= 40 & y_loc > 53.33 & y_loc <= 80] <- 3
fmd19$Dis_location[x_loc >40 & x_loc <= 80 & y_loc <= 26.67] <- 4
fmd19$Dis_location[x_loc >40 & x_loc <= 80 & y_loc > 26.67 & y_loc <= 53.33] <- 5
fmd19$Dis_location[x_loc >40 & x_loc <= 80 & y_loc > 53.33 & y_loc <= 80] <- 6
fmd19$Dis_location[x_loc >80 & x_loc <= 120 & y_loc <= 26.67] <- 7
fmd19$Dis_location[x_loc >80 & x_loc <= 120 & y_loc > 26.67 & y_loc <= 53.33] <- 8
fmd19$Dis_location[x_loc >80 & x_loc <= 120 & y_loc > 53.33 & y_loc <= 80] <- 9

# Manually include a scoreline in the matches

fmd19$shot.outcome.name[is.na(fmd19$shot.outcome.name)] <- "Not Goal"

# Extract just what's needed
temp <- fmd19[, c("match_id", "possession_team.name", "shot.outcome.name")]

# Preallocate result vector
scoreline <- integer(nrow(temp))

# Get unique matches
match_ids <- unique(temp$match_id)

for (match_id in match_ids) {
  idx <- which(temp$match_id == match_id)
  teams <- unique(temp$possession_team.name[idx])
  goals <- setNames(c(0, 0), teams)
  
  for (j in seq_along(idx)) {
    i <- idx[j]
    
    team <- temp$possession_team.name[i]
    opponent <- setdiff(teams, team)
    
    if (!is.na(temp$shot.outcome.name[i]) && temp$shot.outcome.name[i] == "Goal") {
      goals[[team]] <- goals[[team]] + 1
    }
    
    scoreline[i] <- goals[[team]] - goals[[opponent]]
  }
}

# Assign back to full dataframe

fmd19$scoreline <- scoreline

#
#
#
#
#

# Turn into dataframe and remove unused action types

fm19 <- as.data.frame(fmd19)

fm19 <- fm19[which(fm19$type.name != "Starting XI" & fm19$type.name != "Half Start" & 
fm19$type.name != "Half End" & fm19$type.name != "Tactical Shift" & fm19$type.name != "Injury Stoppage"& 
fm19$type.name != "Substitution" & fm19$type.name != "Player Off"& 
fm19$type.name != "Player On" & fm19$type.name != "Camera On"& 
fm19$type.name != "Goal Keeper" & fm19$type.name != "Bad Behaviour"& 
fm19$type.name != "Camera off" & fm19$type.name != "Pressure" & 
fm19$type.name != "Ball Receipt*" & fm19$type.name != "Block" & 
fm19$type.name != "Dribbled Past" & fm19$type.name != "Foul Won" &
fm19$type.name != "Foul Committed" & fm19$type.name != "Error" &
fm19$type.name != "Offside" & fm19$type.name != "Referee Ball-Drop" &
fm19$type.name != "Duel" & fm19$type.name != "Own Goal Against" &
fm19$type.name != "50/50" & fm19$type.name != "Own Goal For"),]

# Create inter-event times after removing unused actions

fm19$timeinseconds <- (fm19$minute*60+fm19$second)
fm19$time <- fm19 $timestamp
fm19$time <- 3600*as.numeric(substr(fm19$time,1,2))+60*as.numeric(substr(fm19$time,4,5))+as.numeric(substr(fm19$time,7,12)) # convert to seconds (down to nearest millisecond)
fm19$diff.time <- c(NA,diff(fm19$time))

fm19$diff.time[which(fm19$diff.time < 0)] <- 0
fm19$diff.time[which(is.na(fm19$diff.time))] <- 0

fm19 <- fm19 %>%
    group_by(match_id) %>%
  mutate(
    possession_change = replace_na(team.id != lag(team.id), TRUE),
    possession     = cumsum(possession_change)
  ) %>%
  ungroup() 


#
#
#
#
#

### Formatting data into data expected for specified model ###

fm19 <- as.data.frame(fm19)

fm19.format <- fm19[,c("competition_id","team.id","match_id","possession","type.name",
"diff.time","Dis_location","possession_change")]

colnames(fm19.format) <- c("comp","TID","MID","PID","act","deltaT",
"zone","possession_change")

#
#
#
#
#

# fm20 set up

fmd20$Dis_location <- fmd20$location

# Extract x (first coordinate), treating NULLs as NA
x_loc <- unlist(lapply(fmd20$location, function(coord) {
  if (is.null(coord) || length(coord) < 1) NA else coord[1]
}))

# Extract y (second coordinate), treating NULLs as NA
y_loc <- unlist(lapply(fmd20$location, function(coord) {
  if (is.null(coord) || length(coord) < 2) NA else coord[2]
}))


fmd20$Dis_location[x_loc <= 40 & y_loc <= 26.67] <- 1
fmd20$Dis_location[x_loc <= 40 & y_loc > 26.67 & y_loc <= 53.33] <- 2
fmd20$Dis_location[x_loc <= 40 & y_loc > 53.33 & y_loc <= 80.5] <- 3
fmd20$Dis_location[x_loc >40 & x_loc <= 80 & y_loc <= 26.67] <- 4
fmd20$Dis_location[x_loc >40 & x_loc <= 80 & y_loc > 26.67 & y_loc <= 53.33] <- 5
fmd20$Dis_location[x_loc >40 & x_loc <= 80 & y_loc > 53.33 & y_loc <= 80.5] <- 6
fmd20$Dis_location[x_loc >80 & x_loc <= 120.5 & y_loc <= 26.67] <- 7
fmd20$Dis_location[x_loc >80 & x_loc <= 120.5 & y_loc > 26.67 & y_loc <= 53.33] <- 8
fmd20$Dis_location[x_loc >80 & x_loc <= 120.5 & y_loc > 53.33 & y_loc <= 80.5] <- 9

#Position: discrete, the unique token related to the players position
# numeric values 1:25 with definitions provided in Open Data Events pdf

#Goal difference: discrete on a +/- scale with + indicating being up and - 
# indicating you are down by goals

fmd20$shot.outcome.name[is.na(fmd20$shot.outcome.name)] <- "Not Goal"

# Extract just what's needed
temp <- fmd20[, c("match_id", "possession_team.name", "shot.outcome.name")]

# Preallocate result vector
scoreline <- integer(nrow(temp))

# Get unique matches
match_ids <- unique(temp$match_id)

for (match_id in match_ids) {
  idx <- which(temp$match_id == match_id)
  teams <- unique(temp$possession_team.name[idx])
  goals <- setNames(c(0, 0), teams)
  
  for (j in seq_along(idx)) {
    i <- idx[j]
    
    team <- temp$possession_team.name[i]
    opponent <- setdiff(teams, team)
    
    if (!is.na(temp$shot.outcome.name[i]) && temp$shot.outcome.name[i] == "Goal") {
      goals[[team]] <- goals[[team]] + 1
    }
    
    scoreline[i] <- goals[[team]] - goals[[opponent]]
  }
}

# Assign back to full dataframe
fmd20$scoreline <- scoreline

#
#
#
#
#

fm20 <- as.data.frame(fmd20)

fm20 <- fm20[which(fm20$type.name != "Starting XI" & fm20$type.name != "Half Start" & 
fm20$type.name != "Half End" & fm20$type.name != "Tactical Shift" & fm20$type.name != "Injury Stoppage"& 
fm20$type.name != "Substitution" & fm20$type.name != "Player Off"& 
fm20$type.name != "Player On" & fm20$type.name != "Camera On"& 
fm20$type.name != "Goal Keeper" & fm20$type.name != "Bad Behaviour"& 
fm20$type.name != "Camera off" & fm20$type.name != "Pressure" & 
fm20$type.name != "Ball Receipt*" & fm20$type.name != "Block" & 
fm20$type.name != "Dribbled Past" & fm20$type.name != "Foul Won" &
fm20$type.name != "Foul Committed" & fm20$type.name != "Error" &
fm20$type.name != "Offside" & fm20$type.name != "Referee Ball-Drop" &
fm20$type.name != "Duel" & fm20$type.name != "Own Goal Against" &
fm20$type.name != "50/50" & fm20$type.name != "Own Goal For"),]

### Need to create interarrival times after removing Unimportant actions ###

fm20$timeinseconds <- (fm20$minute*60+fm20$second)
fm20$time <- fm20 $timestamp
fm20$time <- 3600*as.numeric(substr(fm20$time,1,2))+60*as.numeric(substr(fm20$time,4,5))+as.numeric(substr(fm20$time,7,12)) # convert to seconds (down to nearest millisecond)
fm20$diff.time <- c(NA,diff(fm20$time))

fm20$diff.time[which(fm20$diff.time < 0)] <- 0
fm20$diff.time[which(is.na(fm20$diff.time))] <- 0

fm20 <- fm20 %>%
  # choose the ordering that exists in YOUR data:
  group_by(match_id) %>%
  mutate(
    possession_change = replace_na(team.id != lag(team.id), TRUE),
    possession     = cumsum(possession_change)
  ) %>%
  ungroup() 

### Take the columns that we believe to be useful going forward ###

fm20 <- as.data.frame(fm20)

fm20.format <- fm20[,c("competition_id","team.id","match_id","possession","type.name",
"diff.time","Dis_location","possession_change")]

colnames(fm20.format) <- c("comp","TID","MID","PID","act","deltaT",
"zone","possession_change")

#
#
#
#
#

# fm21 set up

fmd21$Dis_location <- fmd21$location

# Extract x (first coordinate), treating NULLs as NA
x_loc <- unlist(lapply(fmd21$location, function(coord) {
  if (is.null(coord) || length(coord) < 1) NA else coord[1]
}))

# Extract y (second coordinate), treating NULLs as NA
y_loc <- unlist(lapply(fmd21$location, function(coord) {
  if (is.null(coord) || length(coord) < 2) NA else coord[2]
}))


fmd21$Dis_location[x_loc <= 40 & y_loc <= 26.67] <- 1
fmd21$Dis_location[x_loc <= 40 & y_loc > 26.67 & y_loc <= 53.33] <- 2
fmd21$Dis_location[x_loc <= 40 & y_loc > 53.33 & y_loc <= 80.5] <- 3
fmd21$Dis_location[x_loc >40 & x_loc <= 80 & y_loc <= 26.67] <- 4
fmd21$Dis_location[x_loc >40 & x_loc <= 80 & y_loc > 26.67 & y_loc <= 53.33] <- 5
fmd21$Dis_location[x_loc >40 & x_loc <= 80 & y_loc > 53.33 & y_loc <= 80.5] <- 6
fmd21$Dis_location[x_loc >80 & x_loc <= 120.5 & y_loc <= 26.67] <- 7
fmd21$Dis_location[x_loc >80 & x_loc <= 120.5 & y_loc > 26.67 & y_loc <= 53.33] <- 8
fmd21$Dis_location[x_loc >80 & x_loc <= 120.5 & y_loc > 53.33 & y_loc <= 80.5] <- 9

#Position: discrete, the unique token related to the players position
# numeric values 1:25 with definitions provided in Open Data Events pdf

#Goal difference: discrete on a +/- scale with + indicating being up and - 
# indicating you are down by goals

fmd21$shot.outcome.name[is.na(fmd21$shot.outcome.name)] <- "Not Goal"

# Extract just what's needed

temp <- fmd21[, c("match_id", "possession_team.name", "shot.outcome.name")]

# Preallocate result vector

scoreline <- integer(nrow(temp))

# Get unique matches

match_ids <- unique(temp$match_id)

for (match_id in match_ids) {
  idx <- which(temp$match_id == match_id)
  teams <- unique(temp$possession_team.name[idx])
  goals <- setNames(c(0, 0), teams)
  
  for (j in seq_along(idx)) {
    i <- idx[j]
    
    team <- temp$possession_team.name[i]
    opponent <- setdiff(teams, team)
    
    if (!is.na(temp$shot.outcome.name[i]) && temp$shot.outcome.name[i] == "Goal") {
      goals[[team]] <- goals[[team]] + 1
    }
    
    scoreline[i] <- goals[[team]] - goals[[opponent]]
  }
}

# Assign back to full dataframe

fmd21$scoreline <- scoreline

#
#
#
#
#

fm21 <- as.data.frame(fmd21)

fm21 <- fm21[which(fm21$type.name != "Starting XI" & fm21$type.name != "Half Start" & 
fm21$type.name != "Half End" & fm21$type.name != "Tactical Shift" & fm21$type.name != "Injury Stoppage"& 
fm21$type.name != "Substitution" & fm21$type.name != "Player Off"& 
fm21$type.name != "Player On" & fm21$type.name != "Camera On"& 
fm21$type.name != "Goal Keeper" & fm21$type.name != "Bad Behaviour"& 
fm21$type.name != "Camera off" & fm21$type.name != "Pressure" & 
fm21$type.name != "Ball Receipt*" & fm21$type.name != "Block" & 
fm21$type.name != "Dribbled Past" & fm21$type.name != "Foul Won" &
fm21$type.name != "Foul Committed" & fm21$type.name != "Error" &
fm21$type.name != "Offside" & fm21$type.name != "Referee Ball-Drop" &
fm21$type.name != "Duel" & fm21$type.name != "Own Goal Against" &
fm21$type.name != "50/50" & fm21$type.name != "Own Goal For"),]

### Need to create interarrival times after removing Unimportant actions ###

fm21$timeinseconds <- (fm21$minute*60+fm21$second)
fm21$time <- fm21 $timestamp
fm21$time <- 3600*as.numeric(substr(fm21$time,1,2))+60*as.numeric(substr(fm21$time,4,5))+as.numeric(substr(fm21$time,7,12)) # convert to seconds (down to nearest millisecond)
fm21$diff.time <- c(NA,diff(fm21$time))

fm21$diff.time[which(fm21$diff.time < 0)] <- 0
fm21$diff.time[which(is.na(fm21$diff.time))] <- 0

fm21 <- as.data.frame(fm21)

fm21 <- fm21 %>%
  # choose the ordering that exists in YOUR data:
  group_by(match_id) %>%
  mutate(
    possession_change = replace_na(team.id != lag(team.id), TRUE),
    possession     = cumsum(possession_change)
  ) %>%
  ungroup() 

#
#
#
#
#

### Formatting data into data expected for specified model ###

fm21 <- as.data.frame(fm21)

fm21.format <- fm21[,c("competition_id","team.id","match_id","possession","type.name",
"diff.time","Dis_location","possession_change")]

colnames(fm21.format) <- c("comp","TID","MID","PID","act","deltaT",
"zone","possession_change")

#
#
#
#
#

# Add two new action types based on loss of possession 

home_lookup_19 <- data.frame(
  MID = fm.19$match_id,  # all match IDs in your data
  home_TID = fm.19$home_team.home_team_id
)

home_lookup_20 <- data.frame(
  MID = fm.20$match_id,  # all match IDs in your data
  home_TID = fm.20$home_team.home_team_id
)

home_lookup_21 <- data.frame(
  MID = fm.21$match_id,  # all match IDs in your data
  home_TID = fm.21$home_team.home_team_id
)


fm19.format <- fm19.format %>%
  left_join(home_lookup_19, by = "MID")

fm20.format <- fm20.format %>%
  left_join(home_lookup_20, by = "MID")

fm21.format <- fm21.format %>%
  left_join(home_lookup_21, by = "MID")

fm19.format <- fm19.format %>%
  mutate(Home_Away = ifelse(TID == home_TID, "Home", "Away"))

fm20.format <- fm20.format %>%
  mutate(Home_Away = ifelse(TID == home_TID, "Home", "Away"))

fm21.format <- fm21.format %>%
  mutate(Home_Away = ifelse(TID == home_TID, "Home", "Away"))

add_possession_changes <- function(events_df, group_cols = "MID") {
  if (!"Home_Away" %in% names(events_df)) {
    stop("events_df must contain a 'Home_Away' column")
  }

  events_df <- events_df %>% mutate(.orig = row_number())

  # flag the first event of each new possession
  events_df <- events_df %>%
    group_by(across(all_of(group_cols))) %>%
    arrange(.orig, .by_group = TRUE) %>%
    mutate(
      poss_change_flag = case_when(
        Home_Away != lag(Home_Away) & Home_Away == "Home" ~ 1,
        Home_Away != lag(Home_Away) & Home_Away == "Away" ~ 0,
        TRUE ~ NA_real_
      )
    ) %>%
    ungroup()

  # create PC rows from those flagged events, placing them just before
  pc_rows <- events_df %>%
    filter(!is.na(poss_change_flag)) %>%
    mutate(
      act    = paste0("PC", poss_change_flag),
      PID    = NA_integer_,
      deltaT = 0,
      .orig  = .orig - 0.5   # so it sorts before the event itself
    ) %>%
    select(names(events_df))

  # combine, order, and clean up
  out <- bind_rows(events_df, pc_rows) %>%
    arrange(across(all_of(group_cols)), .orig) %>%
    select(-.orig, -poss_change_flag)

  return(out)
}

fm19.format.PC <- add_possession_changes(fm19.format)
fm20.format.PC <- add_possession_changes(fm20.format)
fm21.format.PC <- add_possession_changes(fm21.format)

fm19.format.PC <- as.data.frame(fm19.format.PC)
fm20.format.PC <- as.data.frame(fm20.format.PC)
fm21.format.PC <- as.data.frame(fm21.format.PC)

# Bind the original seasonal datasets to that we can randomly select matches
# into our training, validation and test datasets

full_fm <- rbind(fm19.format.PC, fm20.format.PC, fm21.format.PC) 

set.seed(1)
Random_matches <- sample(unique(full_fm$MID), length(unique(full_fm$MID)), replace = FALSE)

fmtraining <- full_fm[which(full_fm$MID %in% Random_matches[1:196]),]
fmvalid <- full_fm[which(full_fm$MID %in% Random_matches[197:(196+65)]),]
fmtest <- full_fm[which(full_fm$MID %in% Random_matches[(196+65+1):326]),]

fmtraining$zone <- as.numeric(fmtraining$zone)
fmvalid$zone <- as.numeric(fmvalid$zone)
fmtest$zone <- as.numeric(fmtest$zone)

write.csv(fmtraining , "fmtraining.csv", row.names = FALSE)
write.csv(fmvalid , "fmvalid.csv", row.names = FALSE)
write.csv(fmtest , "fmtest.csv", row.names = FALSE)

# Create data frames without all the extra variables, only include comp, TID, MID,
# PID, act (action), deltaT (interarrival times), x and y (location), and zone

fmtraining_short <- fmtraining[,-c(4,8,9,10)]
fmvalid_short <- fmvalid[,-c(4,8,9,10)]
fmtest_short <- fmtest[,-c(4,8,9,10)]

# Create csv files for use in Python 

write.csv(fmtraining_short , "fmtraining_short.csv", row.names = FALSE)
write.csv(fmvalid_short , "fmvalid_short.csv", row.names = FALSE)
write.csv(fmtest_short , "fmtest_short.csv", row.names = FALSE)
