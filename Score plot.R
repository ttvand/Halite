# Clear the workspace
rm(list=ls())

library(data.table)

topTeams <- 10
teams <- c(
  "Tom Van de Wiele", 
  "Raine Force",
  "convexOptimization",
  "Uninstall LoL",
  "Leukocyte",
  "mzotkiew",
  "(⊙﹏⊙)",
  "KhaVo Dan Gilles Robga Tung",
  "Robiland",
  "Fei Wang"
)

scores <- c(1592.3, 1524.5, 1513.5, 1487.8, 1484.3, 1480.9, 1462.3, 1439.7,
            1435.9, 1434.5)

ranking = data.table(
  `Team Name` = teams,
  Score = scores
)

ranking$`Team Name` <- factor(ranking$`Team Name`,
                              levels = ranking$`Team Name`[order(ranking$Score,
                                                                 decreasing = TRUE)])
ranking$`My Team` <- ranking$`Team Name` == "Tom Van de Wiele"
ranking$ScoreLab <- ranking$Score

p <- ggplot(ranking[1:topTeams,], aes(x = `Team Name`,
                                      y = Score,
                                      fill = `My Team`)) + 
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("grey", "#00DC00"), drop = FALSE) +
  coord_cartesian(ylim=c(1400, 1600)) +
  theme(legend.position="none",
        axis.text.x=element_text(angle = 45, vjust = 0.4, hjust = 0.4)) +
  geom_text(aes(label=ScoreLab), position=position_dodge(width=0.9),
            vjust=-0.25)
print(p)