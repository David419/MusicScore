//
//  main.c
//  find_seq
//
//  Created by David Tao on 13/03/2018.
//  Copyright © 2018 David Tao. All rights reserved.
//

int find_seq(int* sub_array, int sub_len, int* to_search_array, int whole_len, int tolerant_num)
{
    int max_error_num = tolerant_num;
    //int max_error_num = tolerant_rate * sub_len;
    int sub_is_found = 1;
    for (int i = 0; i < whole_len - sub_len + 1; i++)
    {
        int error = 0;
        sub_is_found = 1;
        for (int j = 0; j < sub_len; j++)
        {
            if (sub_array[j] != to_search_array[i+j])
            {
                error++;
            }
            if (error > max_error_num)
            {
                sub_is_found = 0;
                break;
            }
        }
        if (sub_is_found)
            return 1;
    }
    return 0;
}

