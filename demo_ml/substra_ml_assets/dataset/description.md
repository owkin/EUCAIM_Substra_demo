# EUCAIM - FL ML demo data description

### Datasource

The data comes from the demo data attached to [rexposome](https://www.bioconductor.org/packages/release/bioc/html/rexposome.html) R package.

A brief introduction to the data can be seen in its own [vignette](https://www.bioconductor.org/packages/release/bioc/vignettes/rexposome/inst/doc/exposome_data_analysis.html).

### Preparation

The data was prepared as seen in the following R code:

```r
# LOAD DATA DOWNLOADED FROM REXPOSOME GITHUB
x <- read.csv( "~/Downloads/exposures.csv", header = TRUE, stringsAsFactors = FALSE )
y <- read.csv( "~/Downloads/phenotypes.csv", header = TRUE, stringsAsFactors = FALSE )

# MERGE AS A SINGLE DATA.FRAME
z <- merge( x, y, by = "idnum" )

# CREATE RANDOM ORDER
o <- sample( seq( nrow( z ) ) )

# SPLIT IN THREE RANDOM SETS
z3.1 <- z[ o[  1:36  ], ]
z3.2 <- z[ o[ 37:72  ], ]
z3.3 <- z[ o[ 73:109 ], ]

write.csv( z3.1, file = "./data/three_dataseties_scenario/z3_1.csv", quote = FALSE, row.names = FALSE )
write.csv( z3.2, file = "./data/three_dataseties_scenario/z3_2.csv", quote = FALSE, row.names = FALSE )
write.csv( z3.3, file = "./data/three_dataseties_scenario/z3_3.csv", quote = FALSE, row.names = FALSE )

# SPLINT IN TWO RANDOM SETS
z2.1 <- z[ o[  1:54  ], ]
z2.2 <- z[ o[ 55:109 ], ]

write.csv( z2.1, file = "./data/two_datasites_scenario/z2_1.csv", quote = FALSE, row.names = FALSE )
write.csv( z2.2, file = "./data/two_datasites_scenario/z2_2.csv", quote = FALSE, row.names = FALSE )
```

Therefore:

 * There are two possible scenarios: two or three data-sites available
 * Each file contains a half or a third of the full data-sets
 * Each files contains the same variables (aka. columns)
 * Some variables have missing values codded as `NA`