#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <handy.h>
#include <string.h>
#include "helper.h"

int episode;

void print_setting(void) {
    printf("----------------- setting -----------------\n");
    printf("- game\n");
    printf("    window_width:       %d\n", TATE);
    printf("    window_height:      %d\n", YOKO);
    printf("    bar_length:         %d\n", BAR_LENGTH);
    printf("\n");
    printf("- reinforcement learning\n");
    printf("    alpha:              %f\n", ALPHA);
    printf("    epsilon:            %f\n", EPSILON);
    printf("    gamma:              %f\n", GAMMA);
    printf("    episode_no:         %d\n", EPISODE_NO);
    printf("    step_no:            %d\n", STEP_NO);
    printf("\n");
    printf("- neural network\n");
    printf("    nnlimit:            %f\n", NNLIMIT);
    printf("    nnalpha:            %f\n", NNALPHA);
    printf("    hidden_unit_no:     %d\n", MID_UNIT_NO);
    printf("    batch_size:         %d\n", BATCH_SIZE);
    printf("    exp_memory_size:    %d\n", EXP_SIZE);
    printf("    loop_limit:         %d\n", LOOP_LIMIT);
    printf("\n");
    printf("- util\n");
    printf("    seed:               %d\n", SEED);
    printf("-------------------------------------------\n\n");
}

void initW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]) {
    int i, j;
    for (i = 0; i < MID_UNIT_NO; i++) {
        for (j = 0; j < INPUT_UNIT_NO + 1; j++) {
            w_mid[i][j] = genrand_real1() * 2 - 1;
        }
    }
    
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        for (j = 0; j < MID_UNIT_NO + 1; j++) {
            w_out[i][j] = genrand_real1() * 2 - 1;
        }
    }
}

void printW(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]) {
    int i, j;

    printf("###################### printw #########################\n");
    for (i = 0; i < MID_UNIT_NO; i++) {
        for (j = 0; j < INPUT_UNIT_NO + 1; j++) {
            printf("%d -> %d: %lf\n", j, i, w_mid[i][j]);
        }
    }
    printf("-----------------------------------------------\n");
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        for (j = 0; j < MID_UNIT_NO + 1; j++) {
            printf("%d -> %d: %lf\n", j, i, w_out[i][j]);
        }
    }
}

void initS(int s[]) {
    s[BALL_X] = genrand_int32() % YOKO;
    s[BALL_Y] = genrand_int32() % (TATE/2) + (TATE/2);
    s[BALL_VEC_X] = genrand_int32() % (SPEED_LIMIT - 1) + 1; // 1 ~ SPEED_LIMITの速さで生成
    s[BALL_VEC_Y] = 1;
    s[BAR_X] = (YOKO/2 + BAR_LENGTH/2);
    // s[BAR_X] = genrand_int32() % (YOKO-BAR_LENGTH);
}

void setinput(double input[], int s[]) {
    input[BALL_X] = (double)s[BALL_X] / YOKO;
    input[BALL_Y] = (double)s[BALL_Y] / TATE;
    input[BALL_VEC_X] = (double)s[BALL_VEC_X] / SPEED_LIMIT;
    input[BALL_VEC_Y] = (double)s[BALL_VEC_Y] / SPEED_LIMIT;
    input[BAR_X] = (double)s[BAR_X] / YOKO;

    // int i;
    // for (i = 0; i < INPUT_UNIT_NO; i++) {
    //     // input[i] = 0;
    // }
    // input[s[BALL_X]] = 1;
    // input[YOKO + s[BALL_Y]] = 1;
    // input[YOKO + TATE + SPEED_LIMIT + s[BALL_VEC_X]] = 1;
    // input[YOKO + TATE + SPEED_LIMIT*2 + 1 + s[BALL_VEC_Y]] = 1;
    // input[YOKO + TATE + SPEED_LIMIT*2 + 2 + s[BAR_X]] = 1;
}

double sigmoidfunc(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double sigmoiddash(double y) {
    return (y * (1 - y));
}

double tanhfunc(double z) {
    return tanh(z);
}

double tanhdash(double y) {
    return (4 / ((exp(y) + exp(-y)) * (exp(y) + exp(-y))));
}

void calcmidunit(double result[], double input_to_unit[], double w[MID_UNIT_NO][INPUT_UNIT_NO + 1]) {
    int i, j;
    double z;
    
    for (i = 0; i < MID_UNIT_NO; i++) {
        z = 0;
        for (j = 0; j < INPUT_UNIT_NO; j++) {
            z += input_to_unit[j] * w[i][j];
        }
        z += (-1) * w[i][j];
        result[i] = tanhfunc(z);
    }
}

void calcoutunit(double result[], double input_to_unit[], double w[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]) {
    int i, j;
    double z;
    
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        z = 0;
        for (j = 0; j < MID_UNIT_NO; j++) {
            z += input_to_unit[j] * w[i][j];
        }
        z -= w[i][j];
        result[i] = tanhfunc(z);
    }
}

int argmaxQ_a(double Q[OUTPUT_UNIT_NO]) {
    int i;
    int max_a;
    
    max_a = 0;
    for (i = 0; i < ACTION_NO; i++) {
        if (Q[i] > Q[max_a]) {
            max_a = i;
        }
    }
    return (max_a);
}

void getQ(int s[], double Q[OUTPUT_UNIT_NO], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO]) {
    double input[INPUT_UNIT_NO];

    // 入力ニューロン設定
    setinput(input, s);
    
    // 中間層の計算
    calcmidunit(result_mid, input, w_mid);

    // 出力層の計算
    calcoutunit(Q, result_mid, w_out);

}

int pi(int s[], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO], float epsilon) {
    double Q[ACTION_NO];
    int maxQ_index;

    getQ(s, Q, w_mid, w_out, result_mid);

    // printf("###############################\n");
    // int i;
    // for (i = 0; i < ACTION_NO; i++) {
    //     printf("Q_%d = %lf\n", i, Q[i]);
    // }
    // for (i = 0; i < INPUT_UNIT_NO; i++) {
    //     printf("s_%d = %d\n", i, s[i]);
    // }

    // イプシロンの確率でランダムに行動
    if (genrand_real1() < epsilon) {
        maxQ_index = (genrand_int32() % ACTION_NO);
    }
    else {
        // 最大のQ値の行動を返す
        maxQ_index = argmaxQ_a(Q);
    }

    return (maxQ_index);
}

void statetransition(int s[], int a, int s_next[], int* hit) {
    int speed_vec = 0;
    s_next[BALL_X] = s[BALL_X] + s[BALL_VEC_X];
    s_next[BALL_Y] = s[BALL_Y] + s[BALL_VEC_Y];
    s_next[BALL_VEC_X] = s[BALL_VEC_X];
    s_next[BALL_VEC_Y] = s[BALL_VEC_Y];

    if (a == ACTION_LEFT) {
        speed_vec = (-1) * BAR_SPEED;
    }
    else if (a == ACTION_RIGHT) {
        speed_vec = BAR_SPEED;
    }
    else if (a == ACTION_STOP) {
        speed_vec = 0;
    }
    s_next[BAR_X] += speed_vec;

    *hit = 0;


    // バーが範囲外に出た時は戻す
    if (s_next[BAR_X] <= 0) {
        s_next[BAR_X] = 0;
    }
    if (s_next[BAR_X] >= YOKO-BAR_LENGTH) {
        s_next[BAR_X] = YOKO-BAR_LENGTH;
    }

    // ボールが横の壁に当たった時
    if ( (s_next[BALL_X] >= YOKO) || (s_next[BALL_X] <= 0) ) {
        s_next[BALL_X] = s[BALL_X] - s[BALL_VEC_X];
        s_next[BALL_Y] = s[BALL_Y] + s[BALL_VEC_X];
        s_next[BALL_VEC_X] = s[BALL_VEC_X] * (-1);
    }

    // ボールが天上に当たった時
    if (s_next[BALL_Y] >= TATE) {
        s_next[BALL_Y] = s[BALL_Y] - s[BALL_VEC_Y];
        s_next[BALL_X] = s[BALL_X] + s[BALL_VEC_X];
        s_next[BALL_VEC_Y] = s[BALL_VEC_Y] * (-1);
    }
    
    // ボールがバーに当たった時
    if ( (s_next[BALL_Y] <= 1) && (s_next[BALL_Y >= -1]) && (s_next[BALL_X] > s_next[BAR_X]) && (s_next[BALL_X] < s_next[BAR_X] + BAR_LENGTH) ) {
        s_next[BALL_Y] = s[BALL_Y] - s[BALL_VEC_Y];
        s_next[BALL_X] = s[BALL_X] + s[BALL_VEC_X] + speed_vec;
        s_next[BALL_VEC_Y] = s[BALL_VEC_Y] * (-1);

        // バーが動いて跳ね返した方向へ加速
        s_next[BALL_VEC_X] = s[BALL_VEC_X] + speed_vec;

        // 加速したものが速度リミットの上限を超えるならば上限に設定
        if (s_next[BALL_VEC_X] > SPEED_LIMIT) {
            s_next[BALL_VEC_X] = SPEED_LIMIT;
        }
        if (s_next[BALL_VEC_X] < (-1)*SPEED_LIMIT) {
            s_next[BALL_VEC_X] = (-1)*SPEED_LIMIT;
        }
        *hit = 1;
    }
}

double reword(int s_next[], int hit) {
    double r;
    if (s_next[BALL_Y] < 0) {
        r = -1.0;
    }
    else if (hit) {
        r = 1.0;
    }
    else {
        r = 0.0;
    }

    // バーが端によっていなければ加点
    if ( (s_next[BAR_X] > YOKO/10) && (s_next[BAR_X] + BAR_LENGTH < YOKO - YOKO/10) ) {
        r += 0.01;
    }

    return r;
}

void append_batch(int s[], int a, double r, int s_next[], int batch_s[BATCH_SIZE][STATE_NO], int batch_a[BATCH_SIZE], double batch_r[BATCH_SIZE], int batch_s_next[BATCH_SIZE][STATE_NO], int batch_count) {
    int i;
    
    for (i = 0; i < STATE_NO; i++) {
        batch_s[batch_count][i] = s[i];
        batch_s_next[batch_count][i] = s_next[i];
    }
    batch_a[batch_count] = a;
    batch_r[batch_count] = r;
}

void append_exp_memory(int s[], int a, double r, int s_next[], int exp_memory_s[BATCH_SIZE][STATE_NO], int exp_memory_a[BATCH_SIZE], double exp_memory_r[BATCH_SIZE], int exp_memory_s_next[BATCH_SIZE][STATE_NO],  int step) {
    int i;
    int min_r_index;

    // メモリがいっぱいになるまでは順次追加
    if (step < EXP_SIZE) {
        append_batch(s, a, r, s_next, exp_memory_s, exp_memory_a, exp_memory_r, exp_memory_s_next, step);
    }
    // いっぱいになったら, その中で最も悪い経験と今回得た経験を比べてよければ交換する
    else {
        min_r_index = 0;
        for (i = 0; i < BATCH_SIZE; i++) {
            if (exp_memory_r[i] < exp_memory_r[min_r_index]) {
                min_r_index = i;
            }
        }
        if (exp_memory_r[min_r_index] > 0) {
            if (r > 0) {
                min_r_index = genrand_int32() % EXP_SIZE;
                append_batch(s, a, r, s_next, exp_memory_s, exp_memory_a, exp_memory_r, exp_memory_s_next, min_r_index);
            }
        }
        else {
            append_batch(s, a, r, s_next, exp_memory_s, exp_memory_a, exp_memory_r, exp_memory_s_next, min_r_index);
        }
    }
}

void shuffle_batch(int batch_s[BATCH_SIZE][STATE_NO], int batch_a[BATCH_SIZE], double batch_r[BATCH_SIZE], int batch_s_next[BATCH_SIZE][STATE_NO], int length) {
    int i, j;
    int index;
    int tmp_s[STATE_NO];
    int tmp_a;
    double tmp_r;
    int tmp_s_next[STATE_NO];

    
    for (i = 0; i < length; i++) {
        index = genrand_int32() % length;
        for (j = 0; j < STATE_NO; j++) {
            tmp_s[j] = batch_s[index][j];
            tmp_s_next[j] = batch_s_next[index][j];

            batch_s[index][j] = batch_s[i][j];
            batch_s_next[index][j] = batch_s_next[i][j];

            batch_s[i][j] = tmp_s[j];
            batch_s_next[i][j] = tmp_s_next[j];
        }
        tmp_a = batch_a[index];
        tmp_r = batch_r[index];

        batch_a[index] = batch_a[i];
        batch_r[index] = batch_r[i];

        batch_a[i] = tmp_a;
        batch_r[i] = tmp_r;
    }
}

double updateQvalue(double Q, double Qnext, double r) {
    double updateQ = Q + (ALPHA * (r - Q + (GAMMA * Qnext)));
    return (updateQ);

    // double t = r + (GAMMA * Qnext);
    // return (t);
}

void make_teacher_data(int a, double t[OUTPUT_UNIT_NO], double Q[OUTPUT_UNIT_NO], double maxQnext, double r) {
    int i;

    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        if (i == a) {
            t[i] = tanhfunc(updateQvalue(Q[a], maxQnext, r));
        }
        else {
            t[i] = Q[i];
        }
    }
}

void bp_for_outunit(double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO], double unit_err[OUTPUT_UNIT_NO], double Q[OUTPUT_UNIT_NO], double t[OUTPUT_UNIT_NO]) {
    int i, j;

    // 出力層ユニット誤差取得
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        unit_err[i] = (Q[i] - t[i]) * tanhdash(Q[i]);
    }

    // 出力層の学習(OUT -> MID)
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        for (j = 0; j < MID_UNIT_NO; j++) {
            w_out[i][j] -= NNALPHA * result_mid[j] * unit_err[i];
        }
        w_out[i][j] -= NNALPHA * (-1.0) * unit_err[i];
    }
}

void bp_for_midunit(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], double input[OUTPUT_UNIT_NO], double result_mid[MID_UNIT_NO], double out_unit_err[OUTPUT_UNIT_NO]) {
    int i, j, k;
    double unit_err[MID_UNIT_NO];
    double sum_wuerr = 0;

    // 中間層の学習(MID -> INPUT)
    for (i = 0; i < MID_UNIT_NO; i++) {
        sum_wuerr = 0;
        for (j = 0; j < OUTPUT_UNIT_NO; j++) {
            sum_wuerr += w_out[j][i] * out_unit_err[j];
        }
        unit_err[i] = tanhdash(result_mid[i]) * sum_wuerr;

        for (j = 0; j < INPUT_UNIT_NO; j++) {
            w_mid[i][j] -= NNALPHA * input[j] * unit_err[i];
        }
        w_mid[i][j] -= NNALPHA * (-1.0) * unit_err[i];
    }
}

double calc_err(double Q[OUTPUT_UNIT_NO], double t[OUTPUT_UNIT_NO]) {
    int i;
    double err = 0.0;

    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        err += (Q[i] - t[i]) * (Q[i] - t[i]);
    }

    return (err);

}

void progress_bar(int step, double err, double err_limit, int l_step) {
    int i;
    char progress_bar[30] = "";
    int progress;

    progress = (int)(((float)step / EPISODE_NO) * 100);

    strcat(progress_bar, "|");
    for (i = 0; i < (progress/4); i++) {
        strcat(progress_bar, "-");
    }
    for (i = (progress/4); i < 25; i++) {
        strcat(progress_bar, " ");
    }
    strcat(progress_bar, "|");

    if (step == 0 && l_step == 0) {
        printf("-");
    }
    fflush(stdout);
    printf("\r%s %d%% %d/%d err=%.3lf/%.3lf lstep=%04d ", progress_bar, progress, step, EPISODE_NO, err, err_limit, l_step);
}

void learning_units(int batch_s[BATCH_SIZE][STATE_NO], int batch_a[BATCH_SIZE], double batch_r[BATCH_SIZE], int batch_s_next[BATCH_SIZE][STATE_NO], double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1], double result_mid[MID_UNIT_NO], int batch_length) {
    int i;
    int l_step;
    double maxQnext;
    double input[INPUT_UNIT_NO];
    double Q[OUTPUT_UNIT_NO], batch_Qnext[BATCH_SIZE][OUTPUT_UNIT_NO];
    double t[OUTPUT_UNIT_NO];
    double co[OUTPUT_UNIT_NO], cm[MID_UNIT_NO]; // cost for output units error, cost for middle units error
    double next_result_mid[MID_UNIT_NO];
    double err = 0.0;
    double err_sum = 100.00;
    double err_limit;

    shuffle_batch(batch_s, batch_a, batch_r, batch_s_next, batch_length);

    for (i = 0; i < BATCH_SIZE; i++) {
        getQ(batch_s_next[i], batch_Qnext[i], w_mid, w_out, next_result_mid);
    }

    l_step = 0;
    err_limit = NNLIMIT*batch_length;
    while ( (err_sum > err_limit) && (l_step < LOOP_LIMIT) ) {
        err_sum = 0.0;
        for (i = 0; i < BATCH_SIZE; i++) {
            getQ(batch_s[i], Q, w_mid, w_out, result_mid);
            
            // 次状態における最大のQ値
            maxQnext = batch_Qnext[i][argmaxQ_a(batch_Qnext[i])];

            // 教師データ生成
            make_teacher_data(batch_a[i], t, Q, maxQnext, batch_r[i]);

            // 出力層の学習(OUT -> MID)
            bp_for_outunit(w_out, result_mid, co, Q, t);

            setinput(input, batch_s[i]);

            // 中間層の学習(MID -> INPUT)
            bp_for_midunit(w_mid, w_out, input, result_mid, co);

            // 誤差を求める
            err = calc_err(Q, t);
            err_sum += err;
        }
        l_step++;
        progress_bar(episode, err_sum, err_limit, l_step);
        // printf("unit learning err = %lf limit = %lf l_step=%d\n", err_sum, NNLIMIT*batch_length, l_step);
    }
}

void load_model(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]) {
    int i,j;
    char line[LINE_NUM];
    FILE *w_mid_file;
    FILE *w_out_file;

    w_mid_file = fopen("model/weight_hidden_unit.model", "r");
    if (w_mid_file == NULL) {
        printf("cannot open\n");
        exit(1);
    }

    w_out_file = fopen("model/weight_output_unit.model", "r");
    if (w_out_file == NULL) {
      printf("cannot open\n");
      exit(1);
    }

    for (i = 0; i < MID_UNIT_NO; i++) {
        for (j = 0; j < INPUT_UNIT_NO + 1; j++) {
            fgets(line, LINE_NUM, w_mid_file);
            sscanf(line, "%lf", &w_mid[i][j]);
        }
    }
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        for (j = 0; j < MID_UNIT_NO + 1; j++) {
            fgets(line, LINE_NUM, w_out_file);
            sscanf(line, "%lf", &w_out[i][j]);
        }
    }
 
    fclose(w_mid_file);
    fclose(w_out_file);
}

void save_model(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]) {
    int i,j;
    FILE *w_mid_file;
    FILE *w_out_file;

    w_mid_file = fopen("model/weight_hidden_unit.model", "w");
    if (w_mid_file == NULL) {
        printf("cannot open\n");
        exit(1);
    }

    w_out_file = fopen("model/weight_output_unit.model", "w");
    if (w_out_file == NULL) {
      printf("cannot open\n");
      exit(1);
    }

    for (i = 0; i < MID_UNIT_NO; i++) {
        for (j = 0; j < INPUT_UNIT_NO + 1; j++) {
            fprintf(w_mid_file, "%.15lf\n", w_mid[i][j]);
        }
    }
    for (i = 0; i < OUTPUT_UNIT_NO; i++) {
        for (j = 0; j < MID_UNIT_NO + 1; j++) {
            fprintf(w_out_file, "%.15lf\n", w_out[i][j]);
        }
    }

    fclose(w_mid_file);
    fclose(w_out_file);
}

void test(double w_mid[MID_UNIT_NO][INPUT_UNIT_NO + 1], double w_out[OUTPUT_UNIT_NO][MID_UNIT_NO + 1]) {
    int t;
    int s[5], a, s_next[5], a_next;
    double result_mid[MID_UNIT_NO];
    int hit;

    initS(s);
    for (t = 0; t < 2000; t++) {
        a = pi(s, w_mid, w_out, result_mid, 0.0);

        HgClear();
        HgCircleFill(s[BALL_X], s[BALL_Y], 5.0, 0);
        HgBoxFill(s[BAR_X], 0, BAR_LENGTH, 3, 0);
        HgSleep(0.01);

        statetransition(s, a, s_next, &hit);
        
        s[BALL_X] = s_next[BALL_X];
        s[BALL_Y] = s_next[BALL_Y];
        s[BALL_VEC_X] = s_next[BALL_VEC_X];
        s[BALL_VEC_Y] = s_next[BALL_VEC_Y];
        s[BAR_X] = s_next[BAR_X];
        
        if (s[BALL_Y] < 0) {
            break;
        }
    }
}