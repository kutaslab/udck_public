# 1. Norming Data Processing

## Example:

### screened responses, variable length, possibly missing data as a pd.Series

```
item_id
i001_2_an_NA_NA              apple
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA                 NA
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA              apple
i001_2_an_NA_NA              apple
i001_2_an_NA_NA              apple
i001_2_an_NA_NA      apple per day
i001_2_an_NA_NA              apple
i001_2_an_NA_NA              apple
i001_2_an_NA_NA              apple
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA            aspirin
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA    apple every day
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA              apple
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA              apple
i001_2_an_NA_NA     apple everyday
i001_2_an_NA_NA         apple tree
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA              apple
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA        apple a day
i001_2_an_NA_NA              apple
Name: screened, dtype: object
```

### ->  `count_responses(responses)` ->


```
count_dict =
    
    {0: {'NA': 1, 'apple': 30, 'aspirin': 1},
     1: {'None': 13, 'a': 15, 'every': 1, 'everyday': 1, 'per': 1, 'tree': 1},
     2: {'None': 15, 'day': 17}}

```

### -> `count_dict_to_df()` ->

```
count_data = 
         
               0   1   2
    NA         1   0   0
    a          0  15   0
    apple     30   0   0
    aspirin    1   0   0
    day        0   0  17
    every      0   1   0
    everyday   0   1   0
    per        0   1   0
    tree       0   1   0

```


## Tabular occurence matrix: counts by position

The counts data is occurence matrix: strings x position

* Format

    * The row-index lists ((string rep of) all unique tokens appearing anywhere in any response.

    * The columns 0, 1, ... denote word position in the responses

    * The value at row,column [i,j] gives the counts of token i at position j.

* Notes: 

    * the number of columns is fixed by the length of the longest response.

    * 'NA' values, if any should occur only in position 0 (= first), else data coding error

    * Shorter responses (length n) are right-padded with string 'None' out to the length
    of the maximum response. Thus 'None' counts should increase monotonically by position, 
    else data coding error




## This matrix makes it easy to answer questions like these:

* Question: What tokens occurred as the first word of a response?  
  Answer: column 0, non-zero values
  
* Question: Word cloze ... what are the proportions of first word responses?  
  Answer: column 0/ sum of column 0, excluding 'NA', 'None'
  
* Question: sentence constraint  
  Answer: maximum word cloze
  
* Question: first word fan, entropy, semantic coherence?  
  Answer: compute from column 0
  
* Question: first N words, fan, entropy, semantic coherence?  
  Answer: compute from columns 0 to N-1

* Question: entire response set fan, entropy, semantic coherence?  
  Answer: compute from occurence matrix

* Did such and such a token occur anywhere in any response?

