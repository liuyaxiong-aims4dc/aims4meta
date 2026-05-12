#!/usr/bin/env Rscript
# ============================================================
# 两组样本差异代谢物分析（通用版）
# 工具: XCMS 4.x + Wilcoxon  |  离子模式: 正离子 (POS)
#
# 差异化合物的两条检测路径：
#
#   路径A — 组特异性特征（定性差异）
#     条件：一组中 ≥X% 样本有信号，另一组中 ≤Y% 样本有信号
#     逻辑：某化合物在一种药材中普遍存在，在另一种中几乎检测不到
#     输出：exclusive_compounds.csv
#
#   路径B — 定量差异特征（两组都有，但强度差异显著）
#     条件：两组均有信号，|log2FC| > 阈值 且 adj.p < 阈值
#     逻辑：某化合物两种药材都含有，但含量相差 N 倍以上
#     输出：differential_compounds.csv
#
#   注意：路径A的特征会从路径B中排除，避免全零组干扰统计
# ============================================================

# ╔══════════════════════════════════════════════════════════╗
# ║                    可调节参数                             ║
# ╚══════════════════════════════════════════════════════════╝

# ── 路径配置 ─────────────────────────────────────────────────
GROUP_A_DIR  <- "/stor1/aims4dc_data/kushen_sandougen/kushen/"
GROUP_B_DIR  <- "/stor1/aims4dc_data/kushen_sandougen/sandougen/"
OUT_DIR      <- "/stor1/aims4dc_data/kushen_sandougen/xcms_results_non-rules-1/"

# ── 组名（用于图表标签和输出列名）────────────────────────────
GROUP_A_NAME <- "kushen"
GROUP_B_NAME <- "sandougen"

# ── 并行核心数 ───────────────────────────────────────────────
N_CORES <- 48

# ════════════════════════════════════════════════════════════
# XCMS 峰提取参数
# ════════════════════════════════════════════════════════════

# ── 峰检测 centWave ──────────────────────────────────────────
CW_PPM         <- 15          # m/z 容差 (ppm)，仪器精度越高可设越小
CW_PEAKWIDTH   <- c(5, 60)    # 色谱峰宽范围 (s)，覆盖窄峰和宽峰
CW_SNTHRESH    <- 6           # 信噪比阈值，越高假峰越少但灵敏度降低
CW_PREFILTER   <- c(3, 1000)  # 预过滤: c(最少连续扫描数, 最低强度)
CW_NOISE       <- 500         # 背景噪声估计值，低于此值的信号忽略
CW_MZCENTER    <- "wMean"     # m/z 中心计算方式
CW_INTEGRATE   <- 1           # 峰面积积分方式 (1=直接积分, 2=高斯拟合)

# ── 保留时间校正 obiwarp ─────────────────────────────────────
OBI_BINSIZE    <- 0.5         # m/z 分箱大小 (Da)，影响校正精度与速度

# ── 峰对齐 PeakDensity ───────────────────────────────────────
# minFraction 按组计算：某特征只要在任意一组内出现比例达标即保留
# 设为 0 可保留所有特征（包括只在单个样本中出现的），但噪声增多
PD_MINFRACTION <- 0.3         # 某组内至少出现的样本比例 (0~1)
PD_BW          <- 5           # 保留时间聚类带宽 (s)
PD_BINSIZE     <- 0.025       # m/z 分箱大小 (Da)


# ── 缺失值填充 ───────────────────────────────────────────────
FILL_MZ_PPM    <- 15          # 回填时 m/z 窗口最小宽度 (ppm)，对应 ChromPeakAreaParam(minMzWidthPpm)

# ════════════════════════════════════════════════════════════
# 路径A：组特异性特征参数（定性差异）
# 一组"有"、另一组"无"的化合物
# ════════════════════════════════════════════════════════════
#
#   EXCL_PRESENT_FRAC：判定某组"有"该化合物的样本比例下限
#     1.0 = 该组所有样本都必须检测到（严格）
#     0.8 = 该组 80% 以上样本检测到即可（宽松，适合大样本量）
#
#   EXCL_ABSENT_FRAC：判定另一组"无"该化合物的样本比例上限
#     0.0 = 另一组完全没有检测到（严格）
#     0.2 = 另一组最多允许 20% 样本有微弱信号（宽松）
#
EXCL_PRESENT_FRAC <- 0.8      # "有"的比例阈值 (0~1)
EXCL_ABSENT_FRAC  <- 0.0      # "无"的比例上限 (0~1)

# ════════════════════════════════════════════════════════════
# 路径B：定量差异特征参数（两组都有，但含量差异显著）
# ════════════════════════════════════════════════════════════
#
#   FC_THRESHOLD：倍数差异阈值，用 log2() 换算
#     log2(2)  ≈ 1.0  → 2倍差异
#     log2(10) ≈ 3.32 → 10倍差异
#     log2(50) ≈ 5.64 → 50倍差异
#
#   USE_WILCOXON：是否启用 Wilcoxon 统计检验
#     TRUE  → 启用，同时要求 adj.p < PVAL_THRESHOLD（适合样本量较多、数据较规范时）
#     FALSE → 不做统计检验，只按 FC_THRESHOLD + MIN_PRESENT_N 规则筛选
#             适合小样本、组内变异大、不想依赖 p 值的情况
#
#   MIN_PRESENT_FRAC：USE_WILCOXON=FALSE 时生效
#     某组内至少有多少比例的样本信号 > 0，才认为该特征在该组"真实存在"
#     同时两组中位数之比需满足 FC_THRESHOLD
#
#   PVAL_THRESHOLD：USE_WILCOXON=TRUE 时生效，adj.p 阈值（BH法）
#
FC_THRESHOLD   <- log2(2)     # 倍数差异阈值，当前设为 2 倍
USE_WILCOXON   <- FALSE        # TRUE=Wilcoxon检验  FALSE=纯规则筛选
MIN_PRESENT_FRAC <- 0.6       # USE_WILCOXON=FALSE 时：每组至少多少比例的样本有信号 (0~1)
PVAL_THRESHOLD <- 0.05        # USE_WILCOXON=TRUE  时：adj.p 阈值

# ════════════════════════════════════════════════════════════
# 可视化参数
# ════════════════════════════════════════════════════════════
HEATMAP_TOPN   <- 50          # 热图展示路径B中显著性最高的前 N 个特征


# ╔══════════════════════════════════════════════════════════╗
# ║                      分析流程                            ║
# ╚══════════════════════════════════════════════════════════╝

suppressPackageStartupMessages({
  library(xcms)
  library(MsExperiment)
  library(BiocParallel)
  library(ggplot2)
  library(pheatmap)
})

register(MulticoreParam(workers = N_CORES))

# 清理旧结果（保留目录本身，删除其中所有文件和子目录）
if (dir.exists(OUT_DIR)) {
  old_files <- list.files(OUT_DIR, full.names = TRUE, recursive = FALSE)
  unlink(old_files, recursive = TRUE)
  cat("已清理旧结果:", OUT_DIR, "\n")
}
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

ga_files  <- list.files(GROUP_A_DIR, pattern = "\\.mzML$", full.names = TRUE)
gb_files  <- list.files(GROUP_B_DIR, pattern = "\\.mzML$", full.names = TRUE)
all_files <- c(ga_files, gb_files)
groups <- c(rep(GROUP_A_NAME, length(ga_files)),
            rep(GROUP_B_NAME, length(gb_files)))

cat(sprintf("=== %s vs %s 差异分析 ===\n", GROUP_A_NAME, GROUP_B_NAME))
cat(sprintf("%s样本: %d  |  %s样本: %d  |  并行核心: %d\n\n",
            GROUP_A_NAME, length(ga_files), GROUP_B_NAME, length(gb_files), N_CORES))
cat("输出目录:", OUT_DIR, "\n\n")

# ── Step 1: 读取原始数据 ──────────────────────────────────────
cat("[1/5] 读取mzML文件...\n")
pd <- data.frame(
  sample_name  = sub("\\.mzML$", "", basename(all_files)),
  sample_group = groups,
  stringsAsFactors = FALSE
)
raw <- MsExperiment::readMsExperiment(
  spectraFiles = all_files,
  sampleData   = pd
)

# ── Step 2: 峰检测 → RT校正 → 峰对齐 → 缺失值填充 ───────────
cat("[2/5] 峰检测 (centWave)...\n")
cwp <- CentWaveParam(
  ppm         = CW_PPM,
  peakwidth   = CW_PEAKWIDTH,
  snthresh    = CW_SNTHRESH,
  prefilter   = CW_PREFILTER,
  mzCenterFun = CW_MZCENTER,
  integrate   = CW_INTEGRATE,
  noise       = CW_NOISE
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

# 提取特征矩阵（行=特征, 列=样本）
feat_mat <- featureValues(xdata, value = "into", method = "sum")
feat_def <- featureDefinitions(xdata)
colnames(feat_mat) <- pd$sample_name

# 保存完整特征表
write.csv(
  cbind(mzmed = feat_def$mzmed, rtmed_min = feat_def$rtmed / 60, as.data.frame(feat_mat)),
  file.path(OUT_DIR, "feature_table.csv"), row.names = TRUE
)

# ── Step 3 & 4: 两条路径并行处理 ─────────────────────────────
cat("[5/5] 差异特征检测（路径A + 路径B）...\n\n")

ga_idx  <- which(groups == GROUP_A_NAME)
gb_idx  <- which(groups == GROUP_B_NAME)

# ────────────────────────────────────────────────────────────
# 路径A：组特异性特征（定性差异）
#   在原始矩阵（填充前）中判断：NA 或 0 均视为"未检测到"
#   目的：找出一组特有、另一组完全没有（或极少）的化合物
# ────────────────────────────────────────────────────────────
detected <- feat_mat > 0 & !is.na(feat_mat)
ga_frac  <- rowSums(detected[, ga_idx, drop = FALSE]) / length(ga_idx)
gb_frac  <- rowSums(detected[, gb_idx, drop = FALSE]) / length(gb_idx)

# 判定"有"与"无"
ga_present  <- ga_frac >= EXCL_PRESENT_FRAC
gb_present  <- gb_frac >= EXCL_PRESENT_FRAC
ga_absent   <- ga_frac <= EXCL_ABSENT_FRAC
gb_absent   <- gb_frac <= EXCL_ABSENT_FRAC

# A组独有：A"有" & B"无"；B组独有：反之
excl_ga_idx  <- which(ga_present & gb_absent)
excl_gb_idx  <- which(gb_present & ga_absent)

make_excl_df <- function(idx, group_name) {
  if (length(idx) == 0) return(NULL)
  data.frame(feature_id  = rownames(feat_mat)[idx],
             mzmed       = feat_def$mzmed[idx],
             rtmed_min   = feat_def$rtmed[idx] / 60,
             only_in = group_name,   # 该化合物仅存在于此组，另一组中未检测到
             stringsAsFactors = FALSE)
}
excl_df <- rbind(make_excl_df(excl_ga_idx, GROUP_A_NAME),
                 make_excl_df(excl_gb_idx, GROUP_B_NAME))

cat("--- 路径A：组特异性特征 ---\n")
cat(sprintf("  判定标准：有≥%.0f%% 样本检测到 / 无≤%.0f%% 样本检测到\n",
            EXCL_PRESENT_FRAC * 100, EXCL_ABSENT_FRAC * 100))
if (!is.null(excl_df) && nrow(excl_df) > 0) {
  excl_intensity <- feat_mat[excl_df$feature_id, , drop = FALSE]
  excl_out <- cbind(excl_df, as.data.frame(excl_intensity))
  write.csv(excl_out, file.path(OUT_DIR, "exclusive_compounds.csv"), row.names = FALSE)
  cat(sprintf("  %s独有: %d 个特征\n", GROUP_A_NAME, length(excl_ga_idx)))
  cat(sprintf("  %s独有: %d 个特征\n", GROUP_B_NAME, length(excl_gb_idx)))
  cat("  → exclusive_compounds.csv\n\n")
} else {
  cat("  未发现符合条件的组特异性特征\n\n")
}

# ────────────────────────────────────────────────────────────
# 路径B：定量差异特征（两组都有，但含量差异显著）
#   将路径A的特征排除后，对剩余特征做中位数归一化 + log2 变换
#   用 Wilcoxon 秩和检验（非参数，对异常值和小样本更鲁棒）
#   多重检验校正：BH法
#   筛选条件：|log2FC| > FC_THRESHOLD 且 adj.p < PVAL_THRESHOLD
# ────────────────────────────────────────────────────────────
excl_ids     <- c(rownames(feat_mat)[excl_ga_idx],
                  rownames(feat_mat)[excl_gb_idx])
feat_mat_sub <- feat_mat[!rownames(feat_mat) %in% excl_ids, , drop = FALSE]

# 中位数归一化：每个样本除以其非零特征的中位数，消除进样量差异
med_norm <- function(mat) {
  mat[mat == 0] <- NA
  factors <- apply(mat, 2, median, na.rm = TRUE)
  sweep(mat, 2, factors, "/")
}
mat_norm <- med_norm(feat_mat_sub)
mat_norm[is.na(mat_norm)] <- 0
mat_log  <- log2(mat_norm + 1)

ga_idx_sub  <- which(groups == GROUP_A_NAME)
gb_idx_sub  <- which(groups == GROUP_B_NAME)

if (USE_WILCOXON) {
  # ── Wilcoxon 秩和检验 ──────────────────────────────────────
  cat("  Wilcoxon 检验中（", nrow(mat_log), "个特征）...\n")
  wilcox_res <- t(sapply(seq_len(nrow(mat_log)), function(i) {
    x    <- mat_log[i, ga_idx_sub]
    y    <- mat_log[i, gb_idx_sub]
    fc   <- mean(y) - mean(x)
    pval <- tryCatch(wilcox.test(x, y, exact = FALSE)$p.value, error = function(e) NA_real_)
    c(log2FC = fc, P.Value = pval)
  }))
  rownames(wilcox_res) <- rownames(mat_log)
  results            <- as.data.frame(wilcox_res)
  results$adj.P.Val  <- p.adjust(results$P.Value, method = "BH")

} else {
  # ── 纯规则筛选（不计算方差，不依赖 p 值）─────────────────
  # log2FC：两组中位数之差（log空间），反映典型倍数差异
  # MIN_PRESENT_FRAC：每组至少多少比例的样本有信号才参与比较
  cat("  规则筛选中（", nrow(mat_log), "个特征，不做统计检验）...\n")
  rule_res <- t(sapply(seq_len(nrow(mat_log)), function(i) {
    x <- mat_log[i, ga_idx_sub]
    y <- mat_log[i, gb_idx_sub]
    fc <- median(y) - median(x)
    ga_n  <- sum(x > 0)
    gb_n  <- sum(y > 0)
    c(log2FC = fc, P.Value = NA_real_, ga_present = ga_n, gb_present = gb_n)
  }))
  rownames(rule_res) <- rownames(mat_log)
  results            <- as.data.frame(rule_res)
  results$adj.P.Val  <- NA_real_
}

results$mzmed       <- feat_def[rownames(results), "mzmed"]
results$rtmed_min   <- feat_def[rownames(results), "rtmed"] / 60
results$AveExpr     <- rowMeans(mat_log)
results$enriched_in <- ifelse(results$log2FC > 0, GROUP_B_NAME, GROUP_A_NAME)
results$fold_change <- round(2^abs(results$log2FC), 1)

# 筛选差异化合物
if (USE_WILCOXON) {
  keep <- abs(results$log2FC) > FC_THRESHOLD &
          !is.na(results$adj.P.Val) &
          results$adj.P.Val < PVAL_THRESHOLD
  criterion_str <- sprintf("|log2FC| > %.2f (%.0f倍) 且 adj.p < %.2f  [Wilcoxon]",
                           FC_THRESHOLD, 2^FC_THRESHOLD, PVAL_THRESHOLD)
} else {
  keep <- abs(results$log2FC) > FC_THRESHOLD &
          results$ga_present  >= MIN_PRESENT_FRAC * length(ga_idx_sub) &
          results$gb_present  >= MIN_PRESENT_FRAC * length(gb_idx_sub)
  criterion_str <- sprintf("|log2FC| > %.2f (%.0f倍) 且 每组 ≥%.0f%% 样本有信号  [规则筛选]",
                           FC_THRESHOLD, 2^FC_THRESHOLD, MIN_PRESENT_FRAC * 100)
}
diff_compounds <- results[keep, ]
diff_compounds <- diff_compounds[order(abs(diff_compounds$log2FC), decreasing = TRUE), ]

cat("--- 路径B：定量差异特征 ---\n")
cat(sprintf("  判定标准：%s\n", criterion_str))
cat(sprintf("  差异特征总数: %d\n",                                        nrow(diff_compounds)))
cat(sprintf("  %s富集 (log2FC < -%.2f): %d 个\n", GROUP_A_NAME, FC_THRESHOLD, sum(diff_compounds$log2FC < -FC_THRESHOLD)))
cat(sprintf("  %s富集 (log2FC > %.2f): %d 个\n",  GROUP_B_NAME, FC_THRESHOLD, sum(diff_compounds$log2FC >  FC_THRESHOLD)))

out_cols <- c("enriched_in", "mzmed", "rtmed_min", "log2FC", "fold_change",
              "P.Value", "adj.P.Val", "AveExpr")

# 附上各样本原始峰面积（方便查看组内重现性）
intensity_mat <- feat_mat_sub[rownames(diff_compounds), , drop = FALSE]
diff_out <- cbind(diff_compounds[, out_cols], as.data.frame(intensity_mat))
write.csv(diff_out, file.path(OUT_DIR, "differential_compounds.csv"))
write.csv(results[, out_cols], file.path(OUT_DIR, "all_features_stats.csv"))
cat("  → differential_compounds.csv\n\n")

# ── 可视化（交互式 HTML）────────────────────────────────────
suppressPackageStartupMessages(library(plotly))

# PCA 交互图
pca    <- prcomp(t(mat_log), scale. = TRUE)
pca_df <- data.frame(PC1   = pca$x[, 1],
                     PC2   = pca$x[, 2],
                     group = groups,
                     sample = pd$sample_name)
var_exp <- round(summary(pca)$importance[2, 1:2] * 100, 1)

p_pca <- plot_ly(pca_df, x = ~PC1, y = ~PC2, color = ~group,
                 colors = c("#E74C3C", "#3498DB") |> setNames(c(GROUP_A_NAME, GROUP_B_NAME)),
                 text  = ~sample, hovertemplate = "%{text}<extra></extra>",
                 type  = "scatter", mode = "markers", marker = list(size = 10)) %>%
  layout(title = sprintf("PCA: %s vs %s", GROUP_A_NAME, GROUP_B_NAME),
         xaxis = list(title = paste0("PC1 (", var_exp[1], "%)")),
         yaxis = list(title = paste0("PC2 (", var_exp[2], "%)")))
htmlwidgets::saveWidget(p_pca, file.path(OUT_DIR, "pca_plot.html"), selfcontained = TRUE)

# 火山图交互图
# Wilcoxon模式：X=log2FC, Y=-log10(adj.p)
# 规则筛选模式：X=log2FC, Y=AveExpr（无p值，用平均信号强度代替Y轴）
sig_b <- paste0(GROUP_B_NAME, "富集")
sig_a <- paste0(GROUP_A_NAME, "富集")
results$sig <- "ns"
if (USE_WILCOXON) {
  results$sig[results$log2FC >  FC_THRESHOLD & !is.na(results$adj.P.Val) & results$adj.P.Val < PVAL_THRESHOLD] <- sig_b
  results$sig[results$log2FC < -FC_THRESHOLD & !is.na(results$adj.P.Val) & results$adj.P.Val < PVAL_THRESHOLD] <- sig_a
} else {
  results$sig[rownames(results) %in% rownames(diff_compounds) & results$log2FC >  FC_THRESHOLD] <- sig_b
  results$sig[rownames(results) %in% rownames(diff_compounds) & results$log2FC < -FC_THRESHOLD] <- sig_a
}
results$hover <- sprintf("m/z: %.4f<br>RT: %.2f min<br>log2FC: %.2f<br>倍数: %.1f x<br>adj.p: %s",
                         results$mzmed, results$rtmed_min,
                         results$log2FC, results$fold_change,
                         ifelse(is.na(results$adj.P.Val), "N/A", sprintf("%.4f", results$adj.P.Val)))

if (USE_WILCOXON) {
  y_val   <- ~-log10(adj.P.Val)
  y_title <- "-log10(adj.p)"
  vol_shapes <- list(
    list(type="line", x0=FC_THRESHOLD,  x1=FC_THRESHOLD,
         y0=0, y1=1, yref="paper", line=list(dash="dash", color="grey")),
    list(type="line", x0=-FC_THRESHOLD, x1=-FC_THRESHOLD,
         y0=0, y1=1, yref="paper", line=list(dash="dash", color="grey")),
    list(type="line", x0=0, x1=1, xref="paper",
         y0=-log10(PVAL_THRESHOLD), y1=-log10(PVAL_THRESHOLD),
         line=list(dash="dash", color="grey")))
} else {
  y_val   <- ~AveExpr
  y_title <- "平均信号强度 log2(intensity)  [规则筛选模式，无p值]"
  vol_shapes <- list(
    list(type="line", x0=FC_THRESHOLD,  x1=FC_THRESHOLD,
         y0=0, y1=1, yref="paper", line=list(dash="dash", color="grey")),
    list(type="line", x0=-FC_THRESHOLD, x1=-FC_THRESHOLD,
         y0=0, y1=1, yref="paper", line=list(dash="dash", color="grey")))
}

p_vol <- plot_ly(results, x = ~log2FC, y = y_val,
                 color = ~sig,
                 colors = c("ns" = "grey70", "#E74C3C", "#3498DB") |>
                            setNames(c("ns", sig_a, sig_b)),
                 text  = ~hover, hovertemplate = "%{text}<extra></extra>",
                 type  = "scatter", mode = "markers",
                 marker = list(size = 6, opacity = 0.7)) %>%
  layout(title = sprintf("火山图: %s vs %s (路径B)", GROUP_A_NAME, GROUP_B_NAME),
         xaxis = list(title = sprintf("log2(FC)  [正值=%s富集]", GROUP_B_NAME)),
         yaxis = list(title = y_title),
         shapes = vol_shapes)
htmlwidgets::saveWidget(p_vol, file.path(OUT_DIR, "volcano_plot.html"), selfcontained = TRUE)

# 热图（路径B Top N，仍用 pheatmap 输出 PNG）
if (nrow(diff_compounds) >= 2) {
  top_n    <- min(HEATMAP_TOPN, nrow(diff_compounds))
  top_id   <- rownames(diff_compounds)[1:top_n]
  mat_heat <- mat_log[top_id, ]
  rownames(mat_heat) <- sprintf("%.4f_%.2fmin",
                                feat_def[top_id, "mzmed"],
                                feat_def[top_id, "rtmed"] / 60)
  ann_col    <- data.frame(group = groups, row.names = colnames(mat_heat))
  ann_colors <- list(group = c("#E74C3C", "#3498DB") |> setNames(c(GROUP_A_NAME, GROUP_B_NAME)))
  png(file.path(OUT_DIR, "heatmap.png"), width = 9, height = 12, units = "in", res = 150)
  pheatmap(mat_heat, annotation_col = ann_col, annotation_colors = ann_colors,
           scale = "row", show_rownames = TRUE, show_colnames = TRUE,
           fontsize_row = 7, main = paste0("Top ", top_n, " 差异化合物热图 (路径B)"))
  dev.off()
}

# ── 完成 ─────────────────────────────────────────────────────
cat("=== 分析完成 ===\n")
cat("输出文件:\n")
cat("  feature_table.csv          — 全部特征量化矩阵（原始）\n")
cat("  exclusive_compounds.csv    — 路径A：组特异性特征（一组有、另一组无）\n")
cat("  differential_compounds.csv — 路径B：定量差异特征（两组都有但含量差异显著）\n")
cat("  all_features_stats.csv     — 路径B：全部特征的统计结果\n")
cat("  pca_plot.html              — PCA 交互图（悬停显示样本名）\n")
cat("  volcano_plot.html          — 火山图交互图（悬停显示 m/z、RT、倍数、adj.p）\n")
cat("  heatmap.png                — 热图（路径B Top N）\n")
cat("输出目录:", OUT_DIR, "\n")
