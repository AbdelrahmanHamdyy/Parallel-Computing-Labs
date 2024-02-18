#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Missing Input.\n");
        return 1;
    }

    int nrows = atoi(argv[1]);
    int ncols = atoi(argv[2]);
    int size = nrows * ncols;

    if (argc - 3 != size) {
        printf("Invalid number of elements provided.\n");
        return 1;
    }

    int **mat = (int **)malloc(nrows * sizeof(int *));
    for (int i = 0; i < nrows; i++) mat[i] = (int *)malloc(ncols * sizeof(int));

    int k = 3;
    for (int i = 0; i < nrows; i++)
        for (int j = 0; j < ncols; j++)
            mat[i][j] = atoi(argv[k++]);

    int sum = 0;
    for (int j = 0; j < ncols; j++) {
        int columnSum = 0;
        for (int i = 0; i < nrows; i++) {
            int t = mat[i][j];
            while (t > 0) {
                columnSum *= 10;
                t /= 10;
            }
            columnSum += mat[i][j];
        }
        sum += columnSum;
    }
    printf("Result: %d\n", sum);

    for (int i = 0; i < nrows; i++) free(mat[i]);
    free(mat);

    return 0;
}