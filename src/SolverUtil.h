#pragma once


/// (i1,i2,i3) -> (i2,i3,i1)
inline void rotateIndiceFwd(double *out, const double *in, const int n1, const int n2, const int n3) {
#ifdef _OPENMP
#	pragma omp parallel for collapse(2) 
#endif
	for (int i2=0; i2<n2; i2++) {
		for (int i3=0; i3<n3; i3++) {
			for (int i1=0; i1<n1; i1++) {
				out[i2*n3*n1+i3*n1+i1] = in[i1*n2*n3+i2*n3+i3];
			}
		}
	}
}

/// (i1,i2,i3) -> (i3,i1,i2)
inline void rotateIndiceBwd(double *out, const double *in, const int n1, const int n2, const int n3) {
#ifdef _OPENMP
#	pragma omp parallel for collapse(2)
#endif
	for (int i3=0; i3<n3; i3++) {
		for (int i1=0; i1<n1; i1++) {
			for (int i2=0; i2<n2; i2++) {
				out[i3*n1*n2+i1*n2+i2] = in[i1*n2*n3+i2*n3+i3];
			}
		}
	}
}



