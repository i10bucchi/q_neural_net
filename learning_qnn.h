#include "const.h"

// MT.c
extern void init_genrand(unsigned long s);

// helper.c
extern void print_setting(void);
extern void initW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]);
extern void printW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]);
extern void initS(int s[]);
extern void getQ(int s[], double Q[OUTPUT_UNIT_NO], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO]);
extern void setinput(double input[], int s[]);
extern int pi(int s[], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO], float epsilon);
extern void statetransition(int s[], int a, int s_next[], int* hit);
extern double reword(int s_next[], int hit);
extern void append_exp_memory(int s[], int a, double r, int s_next[], int exp_memory_s[BATCH_SIZE][STATE_NO], int exp_memory_a[BATCH_SIZE], double exp_memory_r[BATCH_SIZE], int exp_memory_s_next[BATCH_SIZE][STATE_NO],  int step);
extern void append_batch(int s[], int a, double r, int s_next[], int batch_s[BATCH_SIZE][STATE_NO], int batch_a[BATCH_SIZE], double batch_r[BATCH_SIZE], int batch_s_next[BATCH_SIZE][STATE_NO], int batch_count);
extern void learning_units(int batch_s[BATCH_SIZE][STATE_NO], int batch_a[BATCH_SIZE], double batch_r[BATCH_SIZE], int batch_s_next[BATCH_SIZE][STATE_NO], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO], int batch_length);
extern void load_model(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]);
extern void save_model(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]);
extern void test(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]);