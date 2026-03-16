#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;

typedef struct {
    u8* base;
    u64 size;
    u64 offset;
} mem_arena;

typedef struct {
    u32 rows, cols;
    f32* data;
} matrix;

typedef struct {
    matrix* W0;
    matrix* b0;

    matrix* W1;
    matrix* b1;
} neural_network;

mem_arena arena_create(u64 size);
void arena_destroy(mem_arena* arena);
void* arena_push(mem_arena* arena, u64 size);

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);
void mat_mul(matrix* out, const matrix* A, const matrix* B);
void mat_add(matrix* out, const matrix* A, const matrix* B);
void mat_fill_xavier(matrix* out);
void mat_relu(matrix* out, const matrix* A);
void mat_softmax(matrix* out, const matrix* A);
void mat_transpose(matrix* out, const matrix* A);

neural_network* nncreate(mem_arena* arena);
void nn_train(neural_network* nn, mem_arena* arena, matrix* input, matrix* y_true, f32 lr);
void nn_predict(neural_network* nn, mem_arena* arena, matrix* input);

int main(void) {
    mem_arena arena = arena_create(MiB(512));
    neural_network* nn = nncreate(&arena);

    FILE* img_file = fopen("src/mnist_train_images.bin", "rb");
    FILE* lbl_file = fopen("src/mnist_train_labels.bin", "rb");

    int n = 10000;
    f32* images = arena_push(&arena, n * 784 * sizeof(f32));
    u8*  labels = arena_push(&arena, n * sizeof(u8));

    fread(images, sizeof(f32), n * 784, img_file);
    fread(labels, sizeof(u8),  n,       lbl_file);

    fclose(img_file);
    fclose(lbl_file);

    u64 checkpoint = arena.offset;

    matrix* input  = mat_create(&arena, 784, 1);
    matrix* y_true = mat_create(&arena, 10, 1);

    for (int i = 0; i < n; i++) {
        arena.offset = checkpoint;
        input  = mat_create(&arena, 784, 1);
        y_true = mat_create(&arena, 10, 1);

        for (int j = 0; j < 784; j++)
            input->data[j] = images[i * 784 + j];

        for (int j = 0; j < 10; j++) y_true->data[j] = 0.0f;
        y_true->data[labels[i]] = 1.0f;

        nn_train(nn, &arena, input, y_true, 0.01f);

        printf("Step %d — label: %d\n", i, labels[i]);
    }

    printf("Done. Final prediction:\n");

    input = mat_create(&arena, 784, 1);
    n = 100;
    for (int i = 0; i < n; i++) {
        arena.offset = checkpoint;
        input = mat_create(&arena, 784, 1);
        
        for (int j = 0; j < 784; j++)
            input->data[j] = images[i * 784 + j];

        printf("Actual: %d\n", labels[i]);
        nn_predict(nn, &arena, input);
    }

    return 0;
}

mem_arena arena_create(u64 size) {
    mem_arena arena;
    arena.base = malloc(size);
    arena.size = size;
    arena.offset = 0;
    return arena;
}

void arena_destroy(mem_arena* arena) {
    free(arena->base);
}

void* arena_push(mem_arena* arena, u64 size) {
    void* ptr = arena->base + arena->offset;
    arena->offset += size;
    return ptr;
}

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
    matrix* mat = arena_push(arena, sizeof(matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = arena_push(arena, rows * cols * sizeof(f32));
    return mat;
}

void mat_mul(matrix* out, const matrix* A, const matrix* B) {
    for (u32 i = 0; i < A->rows * B->cols; i++) out->data[i] = 0.0f;

    for (u32 i = 0; i < A->rows; i++) {
        for (u32 j = 0; j < B->cols; j++) {
            for (u32 k = 0; k < A->cols; k++) {
                out->data[i * out->cols + j] +=
                    A->data[i * A->cols + k] *
                    B->data[k * B->cols + j];
            } 
        }
    }
}

void mat_add(matrix* out, const matrix* A, const matrix* B) {
    for (u32 i = 0; i < B->rows * B->cols; i++) {
        out->data[i] = A->data[i] + B->data[i];
    }
}

void mat_fill_xavier(matrix* out) {
    f32 fan_in  = out->cols;
    f32 fan_out = out->rows;
    f32 bound = sqrtf(6.0f / (fan_in + fan_out));
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        f32 r = (f32)rand() / (f32)RAND_MAX;
        out->data[i] = r * bound * 2 - bound;
    }
}

void mat_relu(matrix* out, const matrix* A) {
    for (u32 i = 0; i < A->rows * A->cols; i++) {
        out->data[i] = A->data[i] > 0 ? A->data[i] : 0;
    }
}

void mat_softmax(matrix* out, const matrix* A) {
    f32 sum = 0;
    for (u32 i = 0; i < A->rows * A->cols; i++) {
        sum += expf(A->data[i]);
    }

    for (u32 i = 0; i < A->rows; i++) {
        out->data[i] = expf(A->data[i]) / sum;
    }
}

void mat_transpose(matrix* out, const matrix* A) {
    for (u32 i = 0; i < A->rows; i++) {
        for (u32 j = 0; j < A->cols; j++) {
            out->data[j * out->cols + i] = A->data[i * A->cols + j];
        }
    }
}

neural_network* nncreate(mem_arena* arena) {
    neural_network* nn = arena_push(arena, sizeof(neural_network));
    nn->W0 = mat_create(arena, 16, 784);
    nn->b0 = mat_create(arena, 16, 1);
    nn->W1 = mat_create(arena, 10, 16);
    nn->b1 = mat_create(arena, 10, 1);

    mat_fill_xavier(nn->W0);
    mat_fill_xavier(nn->b0);
    mat_fill_xavier(nn->W1);
    mat_fill_xavier(nn->b1);

    return nn;
}

void nn_train(neural_network* nn, mem_arena* arena, matrix* input, matrix* y_true, f32 lr) {
    // FORWARD PASS
    matrix* Z0 = mat_create(arena, 16, 1);
    mat_mul(Z0, nn->W0, input);
    mat_add(Z0, Z0, nn->b0);

    matrix* A0 = mat_create(arena, 16, 1);
    mat_relu(A0, Z0);

    matrix* Z1 = mat_create(arena, 10, 1);
    mat_mul(Z1, nn->W1, A0);
    mat_add(Z1, Z1, nn->b1);

    matrix* A1 = mat_create(arena, 10, 1);
    mat_softmax(A1, Z1);

    // dL/dZ1
    matrix* dZ1 = mat_create(arena, 10, 1);
    for (u32 i = 0; i < 10; i++) {
        dZ1->data[i] = A1->data[i] - y_true->data[i];
    }

    // W1^T
    matrix* W1_T = mat_create(arena, 16, 10);
    mat_transpose(W1_T, nn->W1);

    // dL/dA0
    matrix* dA0 = mat_create(arena, 16, 1);
    for (int i = 0; i < 16; i++) {
        dA0->data[i] = 0;
    }
    mat_mul(dA0, W1_T, dZ1);

    // Update each weight
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 16; j++) {
            nn->W1->data[i * 16 + j] -= A0->data[j] * dZ1->data[i] * lr;
        }
        nn->b1->data[i] -= dZ1->data[i] * lr;
    }

    //dL/dZ0 = dL * dA0/dZ0
    matrix* dZ0 = mat_create(arena, 16, 1);
    for (int i = 0; i < 16; i++) {
        dZ0->data[i] = dA0->data[i] * (Z0->data[i] > 0 ? 1 : 0);
    }

    // Update each weight and bias for W0, B0
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 784; j++) {
            nn->W0->data[i * 784 + j] -= dZ0->data[i] * input->data[j] * lr;
        }
        nn->b0->data[i] -= dZ0->data[i] * lr;
    }
}

void nn_predict(neural_network* nn, mem_arena* arena, matrix* input) {
    matrix* Z0 = mat_create(arena, 16, 1);
    mat_mul(Z0, nn->W0, input);
    mat_add(Z0, nn->b0, Z0);

    matrix* A0 = mat_create(arena, 16, 1);
    mat_relu(A0, Z0);

    matrix* Z1 = mat_create(arena, 10, 1);
    mat_mul(Z1, nn->W1, A0);
    mat_add(Z1, Z1, nn->b1);

    matrix* A1 = mat_create(arena, 10, 1);
    mat_softmax(A1, Z1);

    printf("Prediction:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d: %.2f\n", i, A1->data[i]);
    }
}