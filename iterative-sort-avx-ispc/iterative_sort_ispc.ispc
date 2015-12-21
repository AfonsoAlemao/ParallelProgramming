export void iterative_sort_ispc(
       uniform float randoms[],
       uniform float results[],
       uniform int start,
       uniform int end)
{
  foreach (i = start ... end - 1) {
    float k = 1.0;
    float tmp = 0.0;
    float n = randoms[i];
    while (abs(k - tmp) > 0.0001) {
      tmp = k;
      k = (k + n / k) / 2;
    }
    results[i] = k;
  }
}