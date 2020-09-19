library(data.table)
library(plotly)

data_path <- file.path("~/Kaggle/Halite/Rule agents/Leaderboard replays",
                       "Raine Force comparison.csv")
data <- fread(data_path)
data[score == 600, score := NA]
data <- data[, best_score:=max(score, na.rm = TRUE), submission_id]
data <- data[best_score >= 1450, ]

fig <- plot_ly(data, x=~game_id, y=~score, color=~team_name, mode="lines")
print(fig)
