#!/usr/bin/env Rscript
# ============================================================
# 多组样本差异代谢物分析（通用版，支持 2~N 组）
# 工具: XCMS 4.x  |  离子模式: 正离子 (POS)
#
# 差异化合物的两条检测路径：
#
#   路径A — 组特异性特征（定性差异）
#     对每个特征，计算各组的检出比例
#     输出：exclusive_compounds.csv
#       - only_in：该特征仅在此组检出率 ≥ EXCL_PRESENT_FRAC
#                  且在所有其他组检出率 ≤ EXCL_ABSENT_FRAC
#
#   路径B — 定量差异特征（多组间含量差异显著）
#     Kruskal-Wallis 检验（整体显著性）
#     + Dunn post-hoc 两两比较（找出哪对组之间有差异）
#     输出：differential_compounds.csv
#       - 每对组合各一列 log2FC 和 adj.p
#       - enriched_in：含量最高的组名
#
#   注意：路径A的特征会从路径B中排除
# ============================================================

# ╔══════════════════════════════════════════════════════════╗
# ║                    可调节参数                             ║
# ╚══════════════════════════════════════════════════════════╝

# ── 分组配置（每组一个条目：组名 = mzML所在目录）────────────
GROUPS <- list(
  kushen    = "/stor1/aims4dc_data/kushen_sandougen/kushen/",
  sandougen = "/stor1/aims4dc_data/kushen_sandougen/sandougen/"
  # 添加更多组：
  # huangqi   = "/stor1/aims4dc_data/.../huangqi/",
  # gancao    = "/stor1/aims4dc_data/.../gancao/",
)

OUT_DIR <- "/stor1/aims4dc_data/kushen_sandougen/xcms_results_multigroup/"

# ── 并行核心数 ───────────────────────────────────────────────
N_CORES <- 48

# ════════════════════════════════════════════════════════════
# XCMS 峰提取参数
# ════════════════════════════════════════════════════════════
CW_PPM         <- 15
CW_PEAKWIDTH   <- c(5, 60)
CW_SNTHRESH    <- 6
CW_PREFILTER   <- c(3, 1000)
CW_NOISE       <- 500
CW_MZCENTER    <- "wMean"
CW_INTEGRATE   <- 1

OBI_BINSIZE    <- 0.5

PD_MINFRACTION <- 0.3
PD_BW          <- 5
PD_BINSIZE     <- 0.025

FILL_MZ_PPM    <- 15

# ════════════════════════════════════════════════════════════
# 路径A：组特异性特征参数
# ════════════════════════════════════════════════════════════
#   EXCL_PRESENT_FRAC：判定某组"有"该化合物的样本比例下限
#   EXCL_ABSENT_FRAC ：判定其他所有组"无"该化合物的样本比例上限
#
EXCL_PRESENT_FRAC <- 0.8
EXCL_ABSENT_FRAC  <- 0.0

# ════════════════════════════════════════════════════════════
# 路径B：定量差异特征参数
# ════════════════════════════════════════════════════════════
#   FC_THRESHOLD    ：两两比较中，log2FC 绝对值阈值
#   USE_KW          ：TRUE = Kruskal-Wallis + Dunn post-hoc
#                     FALSE = 纯规则筛选（不依赖 p 值）
#   MIN_PRESENT_FRAC：USE_KW=FALSE 时，每组至少多少比例样本有信号
#   PVAL_THRESHOLD  ：USE_KW=TRUE 时，Dunn adj.p 阈值（BH法）
#
FC_THRESHOLD     <- log2(2)   # 2倍差异
USE_KW           <- FALSE     # TRUE=统计检验  FALSE=规则筛选
MIN_PRESENT_FRAC <- 0.6
PVAL_THRESHOLD   <- 0.05

# ════════════════════════════════════════════════════════════
# 可视化参数
# ════════════════════════════════════════════════════════════
HEATMAP_TOPN <- 50

# ╔══════════════════════════════════════════════════════════╗
# ║                      分析流程                            ║
# ╚══════════════════════════════════════════════════════════╝

suppressPackageStartupMessages({
  library(xcms)
  library(MsExperiment)
  library(BiocParallel)
  library(pheatmap)
  library(plotly)
})

register(MulticoreParam(workers = N_CORES))

group_names <- names(GROUPS)
n_groups    <- length(group_names)
stopifnot("至少需要 2 个分组" = n_groups >= 2)

# 构建文件列表和分组向量
all_files <- unlist(lapply(group_names, function(g)
  list.files(GROUPS[[g]], pattern = "\\.mzML$", full.names = TRUE)))
groups <- unlist(lapply(group_names, function(g) {
  n <- length(list.files(GROUPS[[g]], pattern = "\\.mzML$"))
  rep(g, n)
}))

cat(sprintf("=== 多组差异分析：%s ===\n", paste(group_names, collapse = " / ")))
for (g in group_names)
  cat(sprintf("  %s: %d 个样本\n", g, sum(groups == g)))
cat(sprintf("  并行核心: %d\n\n", N_CORES))

if (dir.exists(OUT_DIR)) {
  unlink(list.files(OUT_DIR, full.names = TRUE, recursive = FALSE), recursive = TRUE)
  cat("已清理旧结果:", OUT_DIR, "\n")
}
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
cat("输出目录:", OUT_DIR, "\n\n")

# ── Step 1: 读取原始数据 ──────────────────────────────────────
cat("[1/5] 读取mzML文件...\n")
pd <- data.frame(
  sample_name  = sub("\\.mzML$", "", basename(all_files)),
  sample_group = groups,
  stringsAsFactors = FALSE
)
raw <- MsExperiment::readMsExperiment(spectraFiles = all_files, sampleData = pd)

# ── Step 2: 峰检测 → RT校正 → 峰对齐 → 缺失值填充 ───────────
cat("[2/5] 峰检测 (centWave)...\n")
cwp <- CentWaveParam(
  ppm = CW_PPM, peakwidth = CW_PEAKWIDTH, snthresh = CW_SNTHRESH,
  prefilter = CW_PREFILTER, mzCenterFun = CW_MZCENTER,
  integrate = CW_INTEGRATE, noise = CW_NOISE
)
xdata <- findChromPeaks(raw, param = cwp, BPPARAM = MulticoreParam(N_CORES))
cat("  检测到色谱峰:", nrow(chromPeaks(xdata)), "\n")

cat("[3/5] 保留时间校正 (obiwarp) + 峰对齐 (PeakDensity)...\n")
xdata <- adjustRtime(xdata, param = ObiwarpParam(binSize = OBI_BINSIZE))
xdata <- groupChromPeaks(xdata, param = PeakDensityParam(
  sampleGroups = groups,
  minFraction  = PD_MINFRACTION,
  bw           = PD_BW,
  binSize      = PD_BINSIZE
))
cat("  对齐后特征数:", nrow(featureDefinitions(xdata)), "\n")

cat("[4/5] 缺失值填充...\n")
xdata <- fillChromPeaks(xdata,
  param   = ChromPeakAreaParam(minMzWidthPpm = FILL_MZ_PPM),
  BPPARAM = MulticoreParam(N_CORES))

feat_mat <- featureValues(xdata, value = "into", method = "sum")
feat_def <- featureDefinitions(xdata)
colnames(feat_mat) <- pd$sample_name

write.csv(
  cbind(mzmed = feat_def$mzmed, rtmed_min = feat_def$rtmed / 60, as.data.frame(feat_mat)),
  file.path(OUT_DIR, "feature_table.csv"), row.names = TRUE
)

# ── Step 5: 差异特征检测 ──────────────────────────────────────
cat("[5/5] 差异特征检测（路径A + 路径B）...\n\n")

# 各组样本列索引
grp_idx <- lapply(group_names, function(g) which(groups == g))
names(grp_idx) <- group_names

# ────────────────────────────────────────────────────────────
# 路径A：组特异性特征
#   某特征在某组检出率 ≥ EXCL_PRESENT_FRAC，
#   且在其余所有组检出率 ≤ EXCL_ABSENT_FRAC → 该组独有
# ────────────────────────────────────────────────────────────
detected <- feat_mat > 0 & !is.na(feat_mat)

# 各组检出比例矩阵（行=特征, 列=组）
frac_mat <- sapply(grp_idx, function(idx)
  rowSums(detected[, idx, drop = FALSE]) / length(idx))

excl_rows <- lapply(group_names, function(g) {
  present <- frac_mat[, g] >= EXCL_PRESENT_FRAC
  others  <- setdiff(group_names, g)
  absent_all <- rowSums(frac_mat[, others, drop = FALSE] <= EXCL_ABSENT_FRAC) == length(others)
  which(present & absent_all)
})
names(excl_rows) <- group_names

excl_list <- lapply(group_names, function(g) {
  idx <- excl_rows[[g]]
  if (length(idx) == 0) return(NULL)
  data.frame(
    feature_id = rownames(feat_mat)[idx],
    mzmed      = feat_def$mzmed[idx],
    rtmed_min  = feat_def$rtmed[idx] / 60,
    only_in    = g,
    stringsAsFactors = FALSE
  )
})
excl_df <- do.call(rbind, excl_list)

cat("--- 路径A：组特异性特征 ---\n")
cat(sprintf("  判定标准：有≥%.0f%% / 其余组均≤%.0f%%\n",
            EXCL_PRESENT_FRAC * 100, EXCL_ABSENT_FRAC * 100))
if (!is.null(excl_df) && nrow(excl_df) > 0) {
  excl_out <- cbind(excl_df, as.data.frame(feat_mat[excl_df$feature_id, , drop = FALSE]))
  write.csv(excl_out, file.path(OUT_DIR, "exclusive_compounds.csv"), row.names = FALSE)
  for (g in group_names)
    cat(sprintf("  %s独有: %d 个特征\n", g, length(excl_rows[[g]])))
  cat("  → exclusive_compounds.csv\n\n")
} else {
  cat("  未发现符合条件的组特异性特征\n\n")
}

# ────────────────────────────────────────────────────────────
# 路径B：定量差异特征
#   排除路径A特征后，中位数归一化 + log2 变换
#   Kruskal-Wallis 整体检验 → Dunn post-hoc 两两比较
#   输出：每对组合的 log2FC 和 adj.p
# ────────────────────────────────────────────────────────────
excl_ids     <- unlist(lapply(group_names, function(g) rownames(feat_mat)[excl_rows[[g]]]))
feat_mat_sub <- feat_mat[!rownames(feat_mat) %in% excl_ids, , drop = FALSE]

med_norm <- function(mat) {
  mat[mat == 0] <- NA
  sweep(mat, 2, apply(mat, 2, median, na.rm = TRUE), "/")
}
mat_norm <- med_norm(feat_mat_sub)
mat_norm[is.na(mat_norm)] <- 0
mat_log  <- log2(mat_norm + 1)

# 所有两两组合
pairs <- combn(group_names, 2, simplify = FALSE)
pair_labels <- sapply(pairs, function(p) paste(p[1], "vs", p[2]))

if (USE_KW) {
  # ── Kruskal-Wallis + Dunn post-hoc ────────────────────────
  if (!requireNamespace("dunn.test", quietly = TRUE))
    stop("请安装 dunn.test 包: install.packages('dunn.test')")
  cat(sprintf("  Kruskal-Wallis 检验中（%d 个特征）...\n", nrow(mat_log)))

  kw_p <- apply(mat_log, 1, function(x) {
    tryCatch(kruskal.test(x ~ factor(groups))$p.value, error = function(e) NA_real_)
  })
  kw_adj <- p.adjust(kw_p, method = "BH")

  # Dunn 两两比较（仅对 KW 显著的特征）
  kw_sig_idx <- which(!is.na(kw_adj) & kw_adj < PVAL_THRESHOLD)
  cat(sprintf("  KW显著特征: %d 个，进行 Dunn post-hoc...\n", length(kw_sig_idx)))

  dunn_fc  <- matrix(NA_real_, nrow = nrow(mat_log), ncol = length(pairs),
                     dimnames = list(rownames(mat_log), pair_labels))
  dunn_adj <- matrix(NA_real_, nrow = nrow(mat_log), ncol = length(pairs),
                     dimnames = list(rownames(mat_log), pair_labels))

  for (i in kw_sig_idx) {
    x   <- mat_log[i, ]
    res <- dunn.test::dunn.test(x, g = groups, method = "bh", altp = TRUE, kw = FALSE)
    for (j in seq_along(pairs)) {
      p  <- pairs[[j]]
      lbl <- pair_labels[j]
      idx_a <- grp_idx[[p[1]]]; idx_b <- grp_idx[[p[2]]]
      dunn_fc[i, lbl]  <- mean(x[idx_b]) - mean(x[idx_a])
      # dunn.test 输出顺序与 combn 一致
      dunn_adj[i, lbl] <- res$P.adjusted[j]
    }
  }

  results <- data.frame(
    mzmed     = feat_def[rownames(mat_log), "mzmed"],
    rtmed_min = feat_def[rownames(mat_log), "rtmed"] / 60,
    AveExpr   = rowMeans(mat_log),
    kw_p      = kw_p,
    kw_adj_p  = kw_adj,
    row.names = rownames(mat_log)
  )
  for (lbl in pair_labels) {
    results[[paste0("log2FC_", lbl)]]  <- dunn_fc[, lbl]
    results[[paste0("adj.p_",  lbl)]]  <- dunn_adj[, lbl]
  }

  # 筛选：KW 显著 且 至少一对组合满足 FC 阈值
  fc_cols <- paste0("log2FC_", pair_labels)
  any_fc  <- rowSums(abs(results[, fc_cols, drop = FALSE]) > FC_THRESHOLD, na.rm = TRUE) > 0
  keep    <- !is.na(kw_adj) & kw_adj < PVAL_THRESHOLD & any_fc
  criterion_str <- sprintf("KW adj.p < %.2f 且 至少一对 |log2FC| > %.2f (%.0f倍)  [KW+Dunn]",
                           PVAL_THRESHOLD, FC_THRESHOLD, 2^FC_THRESHOLD)

} else {
  # ── 纯规则筛选 ─────────────────────────────────────────────
  cat(sprintf("  规则筛选中（%d 个特征，不做统计检验）...\n", nrow(mat_log)))

  results <- data.frame(
    mzmed     = feat_def[rownames(mat_log), "mzmed"],
    rtmed_min = feat_def[rownames(mat_log), "rtmed"] / 60,
    AveExpr   = rowMeans(mat_log),
    row.names = rownames(mat_log)
  )

  # 各组检出数（在 mat_log 子集上重新算）
  for (g in group_names)
    results[[paste0("n_present_", g)]] <- rowSums(mat_log[, grp_idx[[g]], drop = FALSE] > 0)

  # 两两 FC
  for (j in seq_along(pairs)) {
    p   <- pairs[[j]]; lbl <- pair_labels[j]
    idx_a <- grp_idx[[p[1]]]; idx_b <- grp_idx[[p[2]]]
    results[[paste0("log2FC_", lbl)]] <- apply(mat_log, 1, function(x)
      median(x[idx_b]) - median(x[idx_a]))
  }

  # 筛选：至少一对组合满足 FC 阈值，且两组均有足够检出
  fc_cols <- paste0("log2FC_", pair_labels)
  keep <- rep(FALSE, nrow(results))
  for (j in seq_along(pairs)) {
    p    <- pairs[[j]]; lbl <- pair_labels[j]
    na   <- results[[paste0("n_present_", p[1])]]
    nb   <- results[[paste0("n_present_", p[2])]]
    fc   <- abs(results[[paste0("log2FC_", lbl)]])
    keep <- keep | (fc > FC_THRESHOLD &
                    na >= MIN_PRESENT_FRAC * length(grp_idx[[p[1]]]) &
                    nb >= MIN_PRESENT_FRAC * length(grp_idx[[p[2]]]))
  }
  criterion_str <- sprintf("至少一对组合 |log2FC| > %.2f (%.0f倍) 且 每组 ≥%.0f%% 样本有信号  [规则筛选]",
                           FC_THRESHOLD, 2^FC_THRESHOLD, MIN_PRESENT_FRAC * 100)
}

# 标注含量最高的组
results$enriched_in <- apply(
  sapply(group_names, function(g) rowMeans(mat_log[, grp_idx[[g]], drop = FALSE])),
  1, function(x) group_names[which.max(x)]
)

diff_compounds <- results[keep, ]
diff_compounds <- diff_compounds[order(diff_compounds$AveExpr, decreasing = TRUE), ]

cat("--- 路径B：定量差异特征 ---\n")
cat(sprintf("  判定标准：%s\n", criterion_str))
cat(sprintf("  差异特征总数: %d\n", nrow(diff_compounds)))
for (g in group_names)
  cat(sprintf("  %s富集: %d 个\n", g, sum(diff_compounds$enriched_in == g, na.rm = TRUE)))

intensity_mat <- feat_mat_sub[rownames(diff_compounds), , drop = FALSE]
diff_out <- cbind(diff_compounds, as.data.frame(intensity_mat))
write.csv(diff_out, file.path(OUT_DIR, "differential_compounds.csv"))
write.csv(results,  file.path(OUT_DIR, "all_features_stats.csv"))
cat("  → differential_compounds.csv\n\n")

# ── 可视化 ───────────────────────────────────────────────────

# 调色板（最多支持 12 组）
palette_base <- c("#E74C3C","#3498DB","#2ECC71","#F39C12",
                  "#9B59B6","#1ABC9C","#E67E22","#34495E",
                  "#E91E63","#00BCD4","#8BC34A","#FF5722")
group_colors <- setNames(palette_base[seq_len(n_groups)], group_names)

# PCA 交互图
pca    <- prcomp(t(mat_log), scale. = TRUE)
pca_df <- data.frame(PC1 = pca$x[,1], PC2 = pca$x[,2],
                     group = groups, sample = pd$sample_name)
var_exp <- round(summary(pca)$importance[2, 1:2] * 100, 1)

p_pca <- plot_ly(pca_df, x = ~PC1, y = ~PC2, color = ~group,
                 colors = group_colors,
                 text = ~sample, hovertemplate = "%{text}<extra></extra>",
                 type = "scatter", mode = "markers", marker = list(size = 10)) %>%
  layout(title = sprintf("PCA: %s", paste(group_names, collapse = " / ")),
         xaxis = list(title = paste0("PC1 (", var_exp[1], "%)")),
         yaxis = list(title = paste0("PC2 (", var_exp[2], "%)")))
htmlwidgets::saveWidget(p_pca, file.path(OUT_DIR, "pca_plot.html"), selfcontained = TRUE)

# 热图（路径B Top N）
if (nrow(diff_compounds) >= 2) {
  top_n    <- min(HEATMAP_TOPN, nrow(diff_compounds))
  top_id   <- rownames(diff_compounds)[1:top_n]
  mat_heat <- mat_log[top_id, ]
  rownames(mat_heat) <- sprintf("%.4f_%.2fmin",
                                feat_def[top_id, "mzmed"],
                                feat_def[top_id, "rtmed"] / 60)
  ann_col    <- data.frame(group = groups, row.names = colnames(mat_heat))
  ann_colors <- list(group = group_colors)
  png(file.path(OUT_DIR, "heatmap.png"), width = 9, height = 12, units = "in", res = 150)
  pheatmap(mat_heat, annotation_col = ann_col, annotation_colors = ann_colors,
           scale = "row", show_rownames = TRUE, show_colnames = TRUE,
           fontsize_row = 7, main = paste0("Top ", top_n, " 差异化合物热图 (路径B)"))
  dev.off()
}

# ── 完成 ─────────────────────────────────────────────────────
cat("=== 分析完成 ===\n")
cat("输出文件:\n")
cat("  feature_table.csv          — 全部特征量化矩阵\n")
cat("  exclusive_compounds.csv    — 路径A：各组特异性特征\n")
cat("  differential_compounds.csv — 路径B：定量差异特征（含各对组合FC）\n")
cat("  all_features_stats.csv     — 路径B：全部特征统计结果\n")
cat("  pca_plot.html              — PCA 交互图\n")
cat("  heatmap.png                — Top N 差异化合物热图\n")
cat("输出目录:", OUT_DIR, "\n")
