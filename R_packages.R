# R packages used by the scripts in code/, with the versions the analyses were run with
# (R 4.6.1, Bioconductor 3.23). Run this file once to install anything that is missing.

cran <- c(
  tidyverse  = "2.0.0",
  igraph     = "2.3.3",
  tidygraph  = "1.3.1",
  ggraph     = "2.2.2",
  vegan      = "2.7-5",
  rstatix    = "1.1.0",
  broom      = "1.0.13",
  yardstick  = "1.4.0",
  plotROC    = "2.3.3",
  lsa        = "0.73.4",
  pracma     = "2.4.6",
  VGAM       = "1.1-14",
  MASS       = "7.3-65",
  patchwork  = "1.3.2",
  ggrepel    = "0.9.8",
  ggbeeswarm = "0.7.3",
  circlize   = "0.4.18",
  pals       = "1.10",
  scales     = "1.4.0",
  knitr      = "1.51",
  withr      = "3.0.3"
)

bioconductor <- c(
  ComplexHeatmap         = "2.28.0",
  SpiecEasi              = "2.0.0",
  curatedMetagenomicData =  "3.14.0"
)

missing_cran <- setdiff(names(cran), rownames(installed.packages()))
if (length(missing_cran)) install.packages(missing_cran)

missing_bioc <- setdiff(names(bioconductor), rownames(installed.packages()))
if (length(missing_bioc)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
  BiocManager::install(missing_bioc)
}
