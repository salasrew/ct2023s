#include <stdio.h>

typedef struct
{
    int x;
    int y;
} Point;

int main()
{
    Point A;
    Point *p = &A;
    p->x = 24;
    p->y = 6;
    printf("-> => == === != =/= <= >= <=>");
}