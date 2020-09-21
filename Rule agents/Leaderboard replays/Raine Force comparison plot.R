library(data.table)
library(plotly)

data_path <- file.path("~/Kaggle/Halite/Rule agents/Leaderboard replays",
                       "Raine Force comparison.csv")
data <- fread(data_path)
min_score <- 1350
best_per_team <- 20
max_best_submissions <- 5

data[score == 600, score := NA]
data <- data[, best_score:=max(score, na.rm = TRUE), submission_id]
data <- data[best_score >= min_score, ]
unique_team_names <- unique(data$team_name)
for(team in unique_team_names){
  last_scores <- data[team_name == team & game_id == 0]
  keep_submission_ids <- last_scores[score >= sort(
    last_scores$score, decreasing=TRUE)[max_best_submissions], submission_id]
  keep_ids <- data$team_name != team | data$submission_id %in% keep_submission_ids
  table(keep_ids)
  # browser()
  data <- data[keep_ids]
}

fig <- plot_ly(data, x=~game_id, y=~score, color=~team_name, mode="lines")
print(fig)
