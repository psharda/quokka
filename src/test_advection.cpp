//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_advection.cpp
/// \brief Defines a test problem for linear advection.
///

#include "test_advection.hpp"
#include "AMReX_Algorithm.H"
#include "AMReX_BoxArray.H"
#include "AMReX_DistributionMapping.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParallelDescriptor.H"
#include "AdvectionSimulation.hpp"

auto main(int argc, char **argv) -> int
{
	// Initialization
	// (copied from ExaWind)

	amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []() {
		amrex::ParmParse pp("amrex");
		// Set the defaults so that we throw an exception instead of attempting
		// to generate backtrace files. However, if the user has explicitly set
		// these options in their input files respect those settings.
		if (!pp.contains("throw_exception")) {
			pp.add("throw_exception", 1);
		}
		if (!pp.contains("signal_handling")) {
			pp.add("signal_handling", 0);
		}
	});

	int result = 0;

	{ // objects must be destroyed before amrex::finalize, so enter new
	  // scope here to do that automatically

		result = testproblem_advection();

	} // destructors must be called before amrex::Finalize()
	amrex::Finalize();

	return result;
}

struct SawtoothProblem {
};

template <> void AdvectionSimulation<SawtoothProblem>::setInitialConditions()
{
	for (amrex::MFIter iter(state_old_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox(); // excludes ghost zones
		auto const &state = state_new_.array(iter);
		auto const nx = nx_; // class members are not automatically transferred to device!

		amrex::ParallelFor(indexRange, ncomp_, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
			auto value = static_cast<double>((i + nx / 2) % nx) / nx;
			state(i, j, k, n) = value;
		});
	}

	// set flag
	areInitialConditionsDefined_ = true;
}

void ComputeExactSolution(amrex::Array4<amrex::Real> const &exact_arr, amrex::Box const &indexRange,
			const int nvars, const int nx)
{
	amrex::ParallelFor(indexRange, nvars, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
		auto value = static_cast<double>((i + nx / 2) % nx) / nx;
		exact_arr(i, j, k, n) = value;
	});
}

auto testproblem_advection() -> int
{
	// Problem parameters
	const double max_dt = 1e-4;

	// Problem initialization
	AdvectionSimulation<SawtoothProblem> sim;
	sim.maxDt_ = max_dt;

	// set initial conditions
	sim.setInitialConditions();

	// run simulation
	sim.evolve();

	// Compute reference solution
	amrex::MultiFab state_exact(sim.simBoxArray_, sim.simDistributionMapping_, sim.ncomp_,
				    sim.nghost_);

	for (amrex::MFIter iter(sim.state_new_); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &stateExact = state_exact.array(iter);
		ComputeExactSolution(stateExact, indexRange, sim.ncomp_, sim.nx_);
	}

	// Compute error norm
	amrex::MultiFab residual(sim.simBoxArray_, sim.simDistributionMapping_, sim.ncomp_,
				 sim.nghost_);
	amrex::MultiFab::Copy(residual, state_exact, 0, 0, sim.ncomp_, sim.nghost_);
	amrex::MultiFab::Saxpy(residual, -1., sim.state_new_, 0, 0, sim.ncomp_, sim.nghost_);
	
	double min_rel_error = std::numeric_limits<double>::max();
	for(int comp = 0; comp < sim.ncomp_; ++comp) {
		const auto sol_norm = state_exact.norm1(comp);
		const auto err_norm = residual.norm1(comp);
		const double rel_error = err_norm / sol_norm;
		min_rel_error = std::min(min_rel_error, rel_error);
		amrex::Print() << "Relative L1 error norm (comp " << comp << ") = " << rel_error << "\n";
	}

	const double err_tol = 0.015;
	int status = 0;
	if (min_rel_error > err_tol) {
		status = 1;
	}

	return status;
}
