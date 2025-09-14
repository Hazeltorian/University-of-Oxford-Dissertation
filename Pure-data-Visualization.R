# Libraries

library(dplyr)

# Histogram visualization of Possession Sequences

Visual_fm <- rbind(fmtraining_short, fmvalid_short, fmtest_short)

Visual_fm <- Visual_fm %>%
  group_by(MID, PID) %>%
  mutate(PID_seq = cur_group_id()) %>%
  ungroup()

Visual_fm <- as.data.frame(Visual_fm)

PID_len <- Visual_fm %>%
  group_by(PID_seq) %>%
  summarise(length = n(), .groups = "drop")

PID_len <- as.data.frame(PID_len)

pdf("Posslen.pdf", width = 8, height = 5)
hist(PID_len$length,
ylim = c(0,50000), xlim = c(0,200),
xlab = "Length of Possession",
main = "")
dev.off()

# Barplot visualization of number of occurrences for each event in the data

Act_len <- Visual_fm %>%
  group_by(act) %>%
  summarise(length = n())

Act_len <- as.data.frame(Act_len)

barplot(sort(Act_len$length, decreasing = TRUE),
ylim = c(0,300000),
main = "Number of Occurrences \n of Each Event Type", ylab = "",
names.arg = c("Pass","Carry","Ball Recovery","Clearance","Dribble",
"Miscontrol","Dispossessed","Shot","Interception","Shield"),
las = 2, cex.names = 0.7)

#

hist(as.numeric(full_fm$zone[which(full_fm$act == "Carry")]))


