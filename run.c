#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
// Structures

typedef struct {
  int dim; // transformer dim
  int hidden_dim; // for ffn layers
  int n_layers; // number of layers
  int n_heads; // num of query heads
  int n_kv_heads; // num of key/value heads (for GQA and MQA whose kv_heads <
                  // query_heads)
  int vocab_size; // vocabulary size
  int seq_len; // max sequence length
} Config;

typedef struct {
  // token embedding weights
  float* token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  float* rms_attn_weight; // (layer, dim)
  float* rms_ffn_weight; // (layer, dim)
  // weights for attention, note dim == n_heads * head_size
  float* wq; // (layer, dim, n_heads * head_size)
  float* wk; // (layer, dim, kv_heads * head_size)
  float* wv; // (layer, dim, kv_heads * head_size)
  float* wo; // (layer, n_heads * head_size, dim)
  // weights for FFN
  float* up; // (layer, hidden_dim, dim)
  float* gate; // (layer, dim, 4*dim)
  float* down; // (layer, 4*dim, dim)
  // final rmsnorm
  float* rms_final_weight; // (dim,)
  // (Optional): classifier weights for the logits, on the last layer
  float* wcls; // (dim, vocab_size)
} TransformerWeights;

typedef struct {
  float* x; // activation
  float* xb;
} RunState;

typedef struct {
  Config config;
  TransformerWeights weights;
  RunState state;
} Transformer;

typedef struct {
  // Add necessary fields for tokenizer
} Tokenizer;

typedef struct {
  // Add necessary fields for sampler
} Sampler;

void memory_map_weights(
    TransformerWeights* w,
    Config* cfg,
    float* w_ptr,
    int tie_embedding) {
  int dim = cfg->dim;
  int head_size = dim / cfg->n_heads;
  size_t n_layers = cfg->n_layers;

  w->token_embedding_table = w_ptr;
  w_ptr += cfg->vocab_size * dim;

  w->rms_attn_weight = w_ptr;
  w_ptr += n_layers * dim;

  // attn
  w->wq = w_ptr;
  w_ptr += n_layers * dim * dim;
  w->wk = w_ptr;
  w_ptr += n_layers * dim * cfg->n_kv_heads * head_size;
  w->wv = w_ptr;
  w_ptr += n_layers * dim * cfg->n_kv_heads * head_size;
  w->wo = w_ptr;
  w_ptr += n_layers * dim * dim;

  // ffn
  w->rms_ffn_weight = w_ptr;
  w_ptr += n_layers * dim;
  w->gate = w_ptr;
  w_ptr += n_layers * dim * cfg->hidden_dim;
  w->down = w_ptr;
  w_ptr += n_layers * dim * cfg->hidden_dim;
  w->up = w_ptr;
  w_ptr += n_layers * dim * cfg->hidden_dim;

  w->rms_final_weight = w_ptr;
  w_ptr += dim;
  w_ptr += cfg->seq_len * head_size /
      2; // skip what used to be freq_cls_real (for RoPE)
  w_ptr += cfg->seq_len * head_size /
      2; // skip what used to be freq_cls_imag(for RoPE)
  w->wcls = tie_embedding ? w->token_embedding_table : w_ptr;
  printf("w_cls weight 4: %f\n", w->wcls[3]);
}

void read_checkpoint(
    const char* checkpoint_path,
    Config* config,
    TransformerWeights* transformer_weight,
    ssize_t* file_size) {
  printf("lxylog\n");
  FILE* file = fopen(checkpoint_path, "rb");
  if (file == NULL) {
    printf("Couldn't open file %s\n", checkpoint_path);
    exit(EXIT_FAILURE);
  }
  printf("lxylog for opening\n");
  // Config* config = malloc(sizeof(Config));
  size_t read = fread(config, sizeof(Config), 1, file);
  printf("lxylog for read size %ld\n", read);
  if (read != 1) {
    printf("Failed to read config from file\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }
  // Print all Config attributes
  printf("Config:\n");
  printf("  dim: %d\n", config->dim);
  printf("  hidden_dim: %d\n", config->hidden_dim);
  printf("  n_layers: %d\n", config->n_layers);
  printf("  n_heads: %d\n", config->n_heads);
  printf("  n_kv_heads: %d\n", config->n_kv_heads);
  printf("  vocab_size: %d\n", config->vocab_size);
  printf("  seq_len: %d\n", config->seq_len);

  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int tie_embedding = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes

  printf("total size %ld\n", *file_size);
  fclose(file);

  int fd = open(checkpoint_path, O_RDONLY);
  if (fd == -1) {
    printf("open failed for %s\n", checkpoint_path);
    exit(EXIT_FAILURE);
  }

  void* mapped_data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped_data == MAP_FAILED) {
    printf("mmap failed!");
    exit(EXIT_FAILURE);
  }

  // float* weight = (float*)((char*)mapped_data + sizeof(Config));
  float* weight = (float*)((char*)mapped_data + sizeof(Config));
  memory_map_weights(transformer_weight, config, weight, tie_embedding);
}

void malloc_run_state(const Config* c, RunState* s) {
  s->x = malloc(c->seq_len * c->dim * sizeof(float));
  s->xb = malloc(c->seq_len * c->dim * sizeof(float));
}

void build_transformer(Transformer* transformer, const char* checkpoint_path) {
  ssize_t file_size;
  read_checkpoint(
      checkpoint_path, &transformer->config, &transformer->weights, &file_size);
  malloc_run_state(&transformer->config, &transformer->state);
}

// Function declarations

void rmsnorm(float* o, float* x, const float* weight, const int size) {
  float ss = 0.f; // mean of squared values
  for (size_t i = 0; i < size; ++i) {
    ss += x[i] * x[i];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.f / sqrtf(ss);
  for (size_t i = 0; i < size; ++i) {
    printf("weight is %f\n", weight[i]);
    printf("ss is %f\n", ss);
    printf("x[i] is %f\n", x[i]);
    printf("o[i] is %f\n", o[i]);
    float val = x[i] * ss * weight[i];
    printf("val is %f\n", val);
    o[i] = val;
    // o[i] = x[i] * ss * weight[i];
  }
}

// Question 2: Implement softmax
void softmax(float* x, int size) {
  // Your implementation here
}

// Question 3: Implement matmul
void matmul(float* xout, float* x, float* w, int n, int d) {
  // Your implementation here
}

// Question 4: Implement forward (partial)
float* forward(Transformer* transformer, int token, int pos) {
  printf("start to write forward logic\n");
  Config p = transformer->config;
  float* x = transformer->state.x;
  TransformerWeights w = transformer->weights;

  int dim = p.dim;

  // token lookup table
  float* embed_token =
      transformer->weights.token_embedding_table + (token * dim);
  memcpy(x, embed_token, dim * sizeof(float));
  printf("embedding finish, first element is %f\n", x[0]);

  int n_layers = transformer->config.n_layers;
  for (int j = 0; j < n_layers; ++j) {
    rmsnorm(
        transformer->state.xb,
        x,
        transformer->weights.rms_attn_weight,
        transformer->config.dim);
    x = transformer->state.xb;
    printf("after rms_attn, first element is %f\n", transformer->state.x[0]);
  }
}

// Question 5: Implement RoPE in forward function
// (This will be part of the forward function above)

// Question 6: Implement sample
int sample(Sampler* sampler, float* logits) {
  // Your implementation here
}

// Question 7: Implement encode
void encode(
    Tokenizer* t,
    char* text,
    int8_t bos,
    int8_t eos,
    int* tokens,
    int* n_tokens) {
  // Your implementation here
}

// Question 8: Implement decode
char* decode(Tokenizer* t, int prev_token, int token) {
  // Your implementation here
}

// Question 9: Implement generate
void generate(
    Transformer* transformer,
    Tokenizer* tokenizer,
    Sampler* sampler,
    char* prompt,
    int steps) {
  // Your implementation here
}

// Question 10: Implement chat
void chat(
    Transformer* transformer,
    Tokenizer* tokenizer,
    Sampler* sampler,
    char* cli_user_prompt,
    char* cli_system_prompt,
    int steps) {
  // Your implementation here
}

// Main function
int main(int argc, char* argv[]) {
  // Initialize structures
  Transformer transformer;
  Tokenizer tokenizer;
  Sampler sampler;

  // Parse command line arguments and set up configuration
  char* checkpoint_path = NULL;
  // ./run stories42M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
  if (argc > 1) {
    checkpoint_path = argv[1];
  }
  printf("checkpoint path %s\n", checkpoint_path);
  build_transformer(&transformer, checkpoint_path);

  // Call generate or chat based on mode
  float* logits = forward(&transformer, 10, 0);
  // Clean up and free memory

  return EXIT_SUCCESS;
}
