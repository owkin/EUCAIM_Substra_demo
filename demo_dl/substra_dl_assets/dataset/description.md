# EUCAIM - FL DL demo data description

This dataset is made of several Chest X-ray images (anterior-posterior). The images were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou.

### Datasource

The data comes from the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) data-set(s) in publicly available from [kaggle](https://www.kaggle.com)

### Preparation

The data was prepared as seen in the following R code:

```r

# FULL LIST OF FILES - TRAIN
trn.nrm.all <- list.files( "./chest_xray/train/NORMAL/" )
trn.pnm.all <- list.files( "./chest_xray/train/PNEUMONIA/" )

# FULL LIST OF FILES - TEST
tst.nrm.all <- list.files( "./chest_xray/test/NORMAL/" )
tst.pnm.all <- list.files( "./chest_xray/test/PNEUMONIA/" )

# SPLIT IN THREE RANDOM SETS
split <- function( files, ngroups = 3 ) {
  n <- length( files ) / ngroups
  files <- sample( files )
  lapply( seq( ngroups ), function( ii ) {
    start <- 1 + ( ii - 1 ) * n
    end   <- n * ii
    if( ii == ngroups & n - floor( n ) != 0 ) {
      end <- n * ii + 1
    }
    return( files[ start: end ] )
  } )
}

trn.nrm.3 <- split( trn.nrm.all, 3 )
trn.pnm.3 <- split( trn.pnm.all, 3 )

tst.nrm.3 <- split( tst.nrm.all, 3 )
tst.pnm.3 <- split( tst.pnm.all, 3 )


save_ids <- function( split_files, filename ) {
  for( ii in seq( length( split_files ) ) ) {
    localname <- paste0( filename, ii, ".csv" )
    data <- data.frame( image_name = split_files[[ ii ]] )
    write.csv( data, file = localname, quote = FALSE, row.names = FALSE )
  }
}

save_ids( trn.nrm.3, "./data_ids/three_dataseties_scenario/train.nrm.3_" )
save_ids( trn.pnm.3, "./data_ids/three_dataseties_scenario/train.pnm.3_" )

save_ids( tst.nrm.3, "./data_ids/three_dataseties_scenario/test.nrm.3_" )
save_ids( tst.pnm.3, "./data_ids/three_dataseties_scenario/test.pnm.3_" )

# SPLINT IN TWO RANDOM SETS
trn.nrm.2 <- split( trn.nrm.all, 2 )
trn.pnm.2 <- split( trn.pnm.all, 2 )

tst.nrm.2 <- split( tst.nrm.all, 2 )
tst.pnm.2 <- split( tst.pnm.all, 2 )

save_ids( trn.nrm.2, "./data_ids/two_datasites_scenario/train.nrm.2_" )
save_ids( trn.pnm.2, "./data_ids/two_datasites_scenario/train.pnm.2_" )

save_ids( tst.nrm.2, "./data_ids/two_datasites_scenario/test.nrm.2_" )
save_ids( tst.pnm.2, "./data_ids/two_datasites_scenario/test.pnm.2_" )
```

Therefore:

 * There are two possible scenarios: two or three data-sites available
 * Each file contains a half or a third of the full data-sets
 * Each files contains the same variable (aka. `image_name`) with the name of the image that is included in the set