#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#define MAX_FLOAT_LEN 1000

struct matrix {
    float * data;
    int height;
    int width;
};

struct matrix * readMatrix(int fd);

int main(int argc, char* argv[]) {

    int fd[2];

    struct matrix * mat_one;
    struct matrix * mat_two;

    float * res;

    if (argc < 3) {
        fprintf(stderr, "Too few arguments.\n");
        exit(-1);
    }

    fd[0] = open(argv[1], O_RDONLY);
    fd[1] = open(argv[2], O_RDONLY);

    mat_one = readMatrix(fd[0]);
    mat_two = readMatrix(fd[1]);

    close(fd[0]);
    close(fd[1]);

    res = (float *) malloc(sizeof(float) * mat_one->height * mat_two->width);

    matmult(res, mat_one->data, mat_two->data,
            mat_one->height, mat_one->width, mat_two->width);

    printMatrix(mat_one->data, mat_one->height, mat_one->width);
    printMatrix(mat_two->data, mat_two->height, mat_two->width);

    printMatrix(res, mat_one->height, mat_two->width);

}

void printMatrix(float * res, int height, int width) {
    int i;
    printf("%d, %d\n", height, width);
    for (i = 0; i < height*width; i++) {
        printf("%f\n", res[i]);
    }
}

struct matrix * readMatrix(int fd) {
    off_t fsize;
    int i = 0;
    int j = 0;
    int k = 0;
    int width= 0;
    int height = 0;
    char * in;
    char buf[MAX_FLOAT_LEN];
    char c = 'a';
    float * res;
    struct stat buff;
    struct matrix * real_res = malloc(sizeof(struct matrix));

    fstat(fd, &buff);
    fsize = buff.st_size;

    in = mmap(NULL, fsize, PROT_READ, MAP_SHARED, fd, 0);

    /* loop through the file until we reach the end */
    while((c = in[i]) != '\0') {
        if (c == ' ' || c == '\n') {
            if (c == ' ') width++;
            if (c == '\n') height++;
        }
        i++;
    }

    res = (float *) malloc((sizeof(float)) * height * width);
    i = 0;
    j = 0;
    k = 0;

    /* loop through the file again, this time filling our matrix */
    while ((c = in[i]) != '\0') {
        buf[k++] = c;
        if (c == ' ' || c == '\n') {
            buf[k] = '\0';
            res[j] = atof(buf);
            k = 0;
            j++;
        }
        i++;
    }

    real_res->width = width;
    real_res->height = height;
    real_res->data = res;
    return real_res;
}

void writeMatrix(float * mat, int fd) {

}

void matmult(float * res, const float * A, const float * B,
             int hA, int wA, int wB) {
    int i, j, k;
    float a, b;
    float c;

    for (i = 0; i < hA; i++) {
        for (j = 0; j < wB; j++) {
            c = 0;
            for (k = 0; k < wA; k++) {
                a = A[i * wA + k];
                b = B[j * wA + k];
                c += a * b;
            }
            res[i * wB + j] = c;
        }
    }
}
