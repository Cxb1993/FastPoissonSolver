

#include "TDMAParabolicSolver3D.h"
#include "SolverUtil.h"


void TDMAParabolicSolver3D::initialize() {
	const int nx = m_ncell[0];
	const int ny = m_ncell[1];
	const int nz = m_ncell[2];
	const double hx = m_hcell[0];
	const double hy = m_hcell[1];
	const double hz = m_hcell[2];

	const int ndof = nx * ny * nz;

	m_rhs = new double[ndof];
	m_sol = new double[ndof];
	for (int dir=0; dir<Ndim; dir++) {
		m_phi[dir] = new double[ndof];
	}

	for (int dir=0; dir<Ndim; dir++) {
		m_adiag[dir] = new double[ndof];
		m_alo[dir] = new double[ndof];
		m_aup[dir] = new double[ndof];
	}
	m_gamma = new double[ndof];

	//
	buildTriDiag(ny, nz, nx, hx, m_adiag[0], m_alo[0], m_aup[0]);
	buildTriDiag(nz, nx, ny, hy, m_adiag[1], m_alo[1], m_aup[1]);
	buildTriDiag(nx, ny, nz, hz, m_adiag[2], m_alo[2], m_aup[2]);
}

int TDMAParabolicSolver3D::solve() {
	const int nx = m_ncell[0];
	const int ny = m_ncell[1];
	const int nz = m_ncell[2];
	//const double hx = m_hcell[0];
	//const double hy = m_hcell[1];
	//const double hz = m_hcell[2];

	const int ndof = nx * ny * nz;

	// 
	rotateIndiceFwd(m_phi[0], m_rhs, nx, ny, nz);
	tridiagSolve(ndof, m_adiag[0], m_alo[0], m_aup[0], m_sol, m_phi[0], m_gamma);

	//
	rotateIndiceFwd(m_phi[1], m_sol, ny, nz, nx);
	tridiagSolve(ndof, m_adiag[1], m_alo[1], m_aup[1], m_sol, m_phi[1], m_gamma);

	// 
	rotateIndiceFwd(m_phi[2], m_sol, nz, nx, ny);
	tridiagSolve(ndof, m_adiag[2], m_alo[2], m_aup[2], m_sol, m_phi[2], m_gamma);

	return 0;
}


