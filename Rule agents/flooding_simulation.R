# Simulate the probability of a base called flooded as a function of a uniform
# distribution of opponent ships
grid_size <- 21
all_num_opp_ships <- 1:100
all_num_opp_ships2 <- all_num_opp_ships**2
all_num_opp_ships3 <- all_num_opp_ships**3
num_cons_opp_ships <- length(all_num_opp_ships)
max_distance <- 4
num_sims <- 1e4
min_threat_scores <- matrix(rep(0, num_sims*num_cons_opp_ships),
                            nrow=num_cons_opp_ships)
min_threat_counts <- matrix(rep(0, num_sims*num_cons_opp_ships),
                            nrow=num_cons_opp_ships)
base_row <- 10
base_col <- 10
for(i in 1:num_cons_opp_ships){
  cat("Num ships", i, "of", num_cons_opp_ships, "\n")
  num_opp_ships <- all_num_opp_ships[i]
  for(j in 1:num_sims){
    positions <- sample(grid_size*grid_size, num_opp_ships, replace=FALSE)
    cols <- positions %% grid_size
    rows <- (positions-cols)/grid_size
    diff_from_base <- (cols != base_col) | (rows != base_row)
    cols <- cols[diff_from_base]
    rows <- rows[diff_from_base]
    
    row_diff <- rows-base_row
    vert_dist <- abs(row_diff)
    south_dist <- ifelse(rows >= base_row, rows-base_row,
                         rows-base_row+grid_size)
    col_diff <- cols-base_col
    horiz_dist <- abs(col_diff)
    east_dist <- ifelse(cols >= base_col, cols-base_col,
                        cols-base_col+grid_size)
    
    dist <- vert_dist+horiz_dist
    
    # Compute the threat scores
    considered_distance_ids <- dist <= 4  # 12 considered squares
    north_threat_ids <- (south_dist[
      considered_distance_ids] > grid_size/2) & (
        vert_dist[considered_distance_ids] >= horiz_dist[
          considered_distance_ids])
    north_threat_score <- sum(1/dist[
      considered_distance_ids][north_threat_ids])
    south_threat_ids <- (south_dist[
      considered_distance_ids] < grid_size/2) & (
        vert_dist[considered_distance_ids] >= horiz_dist[
          considered_distance_ids])
    south_threat_score <- sum(1/dist[
      considered_distance_ids][south_threat_ids])
    east_threat_ids <- (east_dist[
      considered_distance_ids] < grid_size/2) & (
        vert_dist[considered_distance_ids] <= horiz_dist[
          considered_distance_ids])
    east_threat_score <- sum(1/dist[
      considered_distance_ids][east_threat_ids])
    west_threat_ids <- (east_dist[
      considered_distance_ids] > grid_size/2) & (
        vert_dist[considered_distance_ids] <= horiz_dist[
          considered_distance_ids])
    west_threat_score <- sum(1/dist[
      considered_distance_ids][west_threat_ids])
    
    min_threat_counts[i, j] <- min(c(sum(north_threat_ids), sum(south_threat_ids),
                                     sum(east_threat_ids), sum(west_threat_ids)))
    min_threat_scores[i, j] <- min(c(north_threat_score, south_threat_score,
                                     east_threat_score, west_threat_score))
  }
}
# mean_min_threat_counts <- rowMeans(min_threat_counts)
# plot(all_num_opp_ships, mean_min_threat_counts)

mean_min_threat_scores <- rowMeans(min_threat_scores)
linear_mod <- lm(mean_min_threat_scores ~ all_num_opp_ships + all_num_opp_ships2+
                  all_num_opp_ships3)
summary(linear_mod)
plot(all_num_opp_ships, mean_min_threat_scores, xlim=c(0, 200), ylim=c(0, 1))
lines(all_num_opp_ships, predict(linear_mod), col='blue')

predict_ships <- 101:200
new <- data.frame(all_num_opp_ships = predict_ships,
                  all_num_opp_ships2 = predict_ships**2,
                  all_num_opp_ships3 = predict_ships**3
                  )
lines(predict_ships, predict(linear_mod, new), col="green")

