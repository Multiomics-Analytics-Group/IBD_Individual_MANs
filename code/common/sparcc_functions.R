# Code adapted from https://github.com/zdk123/SpiecEasi/blob/master/R/spaRcc.R
# Original code is almost untouched, except for the addition of a detrending step and aggregating steps.

sparcc.modified <- function(data, iter = 20, inner_iter = 10, th = .1, detrend = TRUE,
    aggregate_to_individual = FALSE, individual_mapping = NULL) {
    #  without all the 'frills'
    sparccs <- lapply(1:iter, function(i) {
      # apply + norm_diric transposes the data, so we need to transpose it back for sparccinner
        sparccinner(t(apply(data, 1, norm_diric)),
            iter = inner_iter, th = th, detrend = detrend,
            aggregate_to_individual = aggregate_to_individual,
            individual_mapping = individual_mapping
        )
    })
    # collect
    cors <- array(
        unlist(lapply(sparccs, function(x) x$Cor)),
        c(ncol(data), ncol(data), iter)
    )
    corMed <- apply(cors, 1:2, median)
    covs <- array(
        unlist(lapply(sparccs, function(x) x$Cov)),
        c(ncol(data), ncol(data), iter)
    )
    covMed <- apply(covs, 1:2, median)
    covMed <- SpiecEasi::cor2cov(corMed, sqrt(diag(covMed)))
    list(Cov = covMed, Cor = corMed)
}

sparccinner <- function(data.f, T = NULL, iter = 10, th = 0.1, detrend, aggregate_to_individual, individual_mapping) {
    if (is.null(T)) T <- av(data.f, detrend = detrend,
        aggregate_to_individual = aggregate_to_individual,
        individual_mapping = individual_mapping)
    res.bv <- basis_var(T)
    Vbase <- res.bv$Vbase
    M <- res.bv$M
    cbase <- C_from_V(T, Vbase)
    Cov <- cbase$Cov
    Cor <- cbase$Cor

    ## do iterations here
    excluded <- NULL
    for (i in 1:iter) {
        res.excl <- exclude_pairs(Cor, M, th, excluded)
        M <- res.excl$M
        excluded <- res.excl$excluded
        if (res.excl$break_flag) break
        res.bv <- basis_var(T, M = M, excluded = excluded)
        Vbase <- res.bv$Vbase
        M <- res.bv$M
        K <- M
        diag(K) <- 1
        cbase <- C_from_V(T, Vbase)
        Cov <- cbase$Cov
        Cor <- cbase$Cor
    }
    list(Cov = Cov, Cor = Cor, i = i, M = M, excluded = excluded)
}

exclude_pairs <- function(Cor, M, th = 0.1, excluded = NULL) {
    # exclude pairs with high correlations
    break_flag <- FALSE
    C_temp <- abs(Cor - diag(diag(Cor))) # abs value / remove diagonal
    if (!is.null(excluded)) C_temp[excluded] <- 0 # set previously excluded correlations to 0
    exclude <- which(abs(C_temp - max(C_temp)) < .Machine$double.eps * 100)[1:2]
    if (max(C_temp) > th) {
        i <- na.exclude(arrayInd(exclude, c(nrow(M), ncol(M)))[, 1])
        M[i, i] <- M[i, i] - 1
        excluded_new <- c(excluded, exclude)
    } else {
        excluded_new <- excluded
        break_flag <- TRUE
    }
    list(M = M, excluded = excluded_new, break_flag = break_flag)
}

basis_cov <- function(data.f) {
    # data.f -> relative abundance data
    # OTUs in columns, samples in rows (yes, I know this is transpose of normal)
    # first compute aitchison variation
    T <- av(data.f)
    res.bv <- basis_var(T)
    Vbase <- res.bv$Vbase
    M <- res.bv$M
    cbase <- C_from_V(T, Vbase)
    Cov <- cbase$Cov
    Cor <- cbase$Cor
    list(Cov = Cov, M = M)
}

basis_var <- function(T, CovMat = matrix(0, nrow(T), ncol(T)),
                      M = matrix(1, nrow(T), ncol(T)) + (diag(ncol(T)) * (ncol(T) - 2)),
                      excluded = NULL, Vmin = 1e-4) {
    if (!is.null(excluded)) {
        T[excluded] <- 0
        #   CovMat[excluded] <- 0
    }
    Ti <- matrix(rowSums(T))
    CovVec <- matrix(rowSums(CovMat - diag(diag(CovMat)))) # row sum of off diagonals
    M.I <- tryCatch(solve(M), error = function(e) MASS::ginv(M))
    Vbase <- M.I %*% (Ti + 2 * CovVec)
    Vbase[Vbase < Vmin] <- Vmin
    list(Vbase = Vbase, M = M)
}

C_from_V <- function(T, Vbase) {
    J <- matrix(1, nrow(T), ncol(T))
    Vdiag <- diag(c(Vbase))
    CovMat <- .5 * ((J %*% Vdiag) + (Vdiag %*% J) - T)
    CovMat <- (CovMat + t(CovMat)) / 2 # enforce symmetry
    # check that correlations are within -1,1
    CorMat <- cov2cor(CovMat)
    CorMat[abs(CorMat) > 1] <- sign(CorMat[abs(CorMat) > 1])
    CovMat <- SpiecEasi::cor2cov(CorMat, sqrt(as.vector(Vbase)))
    list(Cov = CovMat, Cor = CorMat)
}

av <- function(data, detrend, aggregate_to_individual, individual_mapping) {
    
    clr_data <- SpiecEasi::clr(data)
  
    if(detrend & aggregate_to_individual) {
        stop("Cannot detrend data if aggregate_to_individual is TRUE. Please set one of these options to FALSE.")
    }

    if(aggregate_to_individual & !is.null(individual_mapping)) {
      
        sample_names <- rownames(clr_data)
        mapping_vector <- individual_mapping[sample_names, 1]

        # ADAPTATION: aggregate to individual level by taking the mean of the CLR values for each individual
        print("Aggregating data to individual level")
        clr_data <- aggregate(clr_data, by = list(subj = mapping_vector), FUN = mean)
        clr_data <- as.matrix(clr_data[, -1]) # remove the grouping column
        rownames(clr_data) <- unique(mapping_vector) # set row names to individual IDs

    } else if(aggregate_to_individual & is.null(individual_mapping)) {
        stop("individual_mapping must be provided if aggregate_to_individual is TRUE.")
    }

    if (detrend) {
        # ADAPTATION: linear detrending BEFORE computing covariance
        print("Detrending data in CLR space")
        clr_data <- pracma::detrend(clr_data, tt = "linear")
    }

    cov.clr <- cov(clr_data)
    J <- matrix(1, ncol(data), ncol(data))
    (J %*% diag(diag(cov.clr))) + (diag(diag(cov.clr)) %*% J) - (2 * cov.clr)
}


norm_diric <- function(x, rep = 1) {
    dmat <- VGAM::rdiric(rep, x + 1)
    SpiecEasi::norm_to_total(colMeans(dmat))
}
