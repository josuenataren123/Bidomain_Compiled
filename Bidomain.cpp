/*
 ============================================================================

 .______    _______     ___   .___________.    __  .___________.
 |   _  \  |   ____|   /   \  |           |   |  | |           |
 |  |_)  | |  |__     /  ^  \ `---|  |----`   |  | `---|  |----`
 |   _  <  |   __|   /  /_\  \    |  |        |  |     |  |
 |  |_)  | |  |____ /  _____  \   |  |        |  |     |  |
 |______/  |_______/__/     \__\  |__|        |__|     |__|

 BeatIt - code for cardiovascular simulations
 Copyright (C) 2016 Simone Rossi

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ============================================================================
 */

#include "libmesh/exodusII_io.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_modification.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/parallel_mesh.h"
#include "libmesh/elem.h"
#include "libmesh/analytic_function.h"

#include "libmesh/equation_systems.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/dof_map.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/quadrature_gauss.h"

#include "libmesh/linear_implicit_system.h"
#include "libmesh/transient_system.h"
#include "libmesh/explicit_system.h"

#include <sys/stat.h>
#include "libmesh/getpot.h"

#include "libmesh/vtk_io.h"
#include <iomanip>      // std::setw

using namespace libMesh;

// Domain numbering
enum class Subdomain : unsigned int
{
    TISSUE = 1, BATH = 2
};
// Time integrators
enum class TimeIntegrator : unsigned int
{
    SBDF1 = 1, SBDF2 = 2, SBDF3 = 3, EXPLICIT_EXTRACELLULAR = 4, EXPLICIT_INTRACELLULAR = 5, SEMI_IMPLICIT = 6, SEMI_IMPLICIT_HALF_STEP = 7
};
enum class EquationType : unsigned int
{
    PARABOLIC = 1, ELLIPTIC = 2
};

// Store here details about time stepping
struct TimeData
{
    // constructor
    TimeData(GetPot &data);
    // show parameters
    void print();
    // parameters
    double time;      // Time
    double end_time;      // Final Time
    double dt;            // Timestep
    int timestep;         // timestep coutner
    int export_timesteps; // interval at which export the solution
};


struct IonicModel
{
    void setup(GetPot& data)
    {
        v0 = data("v0", 0.0);
        v1 = data("v1", 0.1);
        v2 = data("v2", 1.0);
        k = data("k", 8.0);
    }

    double iion(double vn)
    {
        return k * (vn-v0) * (vn-v1) * (vn-v2);
    }

    void print()
    {
        std::cout << "**************************************** " << std::endl;
        std::cout << "Ionic model parameters:" << std::endl;
        std::cout << "k: " << k << std::endl;
        std::cout << "v0: " << v0 << std::endl;
        std::cout << "v1: " << v1 << std::endl;
        std::cout << "v2: " << v2 << std::endl;
        std::cout << "**************************************** " << std::endl;
    }

    double k;
    double v0, v1, v2;
};

struct Pacing
{
    Pacing(GetPot& data)
    {
        duration = data("stimulus_duration", 2.0);
        amplitude = data("stimulus_amplitude", 50.0);
        start_time = data("stimulus_start_time", 0.0);
        double maxx = data("stimulus_maxx", 1.0);
        double maxy = data("stimulus_maxy", 1.0);
        double maxz = data("stimulus_maxz", 1.0);
        double minx = data("stimulus_minx", 0.0);
        double miny = data("stimulus_miny", 0.0);
        double minz = data("stimulus_minz", 0.0);
        box = libMesh::BoundingBox(libMesh::Point(minx, miny, minz), libMesh::Point(maxx, maxy, maxz));
    }

    double istim(const libMesh::Point& p, double time)
    {
        double I = 0.0;
        if(time > start_time && time <= start_time+ duration)
        {
            if(box.contains_point(p))
            {
                I = amplitude;
            }
        }
        return I;
    }

    void print()
    {
        std::cout << "**************************************** " << std::endl;
        std::cout << "Pacing parameters:" << std::endl;
        std::cout << "start_time: " << start_time << std::endl;
        std::cout << "duration: " << duration << std::endl;
        std::cout << "amplitude: " << amplitude << std::endl;
        std::cout << "min: " << std::endl;
        box.min().print();
        std::cout << "\nmax: " << std::endl;
        box.max().print();
        std::cout << "\n**************************************** " << std::endl;
        std::cout << std::endl;
    }

    double duration;
    double amplitude;
    double start_time;
    libMesh::BoundingBox box;
};


double initial_condition_V(const libMesh::Point& p, const double time);
double initial_condition_Ve(const libMesh::Point& p, const double time);
double exact_solution_V(const libMesh::Point& p, const double time);
double exact_solution_Ve(const libMesh::Point& p, const double time);
double exact_solution_Ve_Vb(const libMesh::Point& p, const double time);
void exact_solution_monolithic(libMesh::DenseVector<libMesh::Number>& output, const libMesh::Point& p, const double time);



// Read mesh as specified in the input file
void read_mesh(const GetPot &data, libMesh::ParallelMesh &mesh);
// read BC sidesets from string: e.g. bc = "1 2 3 5", or bc = "1, 55, 2, 33"
void read_bc_list(std::string &bc, std::set<int> &bc_sidesets);
// Assemble matrices
void assemble_matrices(libMesh::EquationSystems &es, const TimeData &timedata, TimeIntegrator time_integrator, libMesh::Order p_order, const GetPot &data);
// Solve ionic model / evaluate ionic currents
void solve_ionic_model_evaluate_ionic_currents(libMesh::EquationSystems &es,
        IonicModel& ionic_model,
        Pacing& pacing,
        TimeData& datatime,
        TimeIntegrator& time_integrator);
// void assemble RHS
void assemble_rhs(  libMesh::EquationSystems &es,
                    const TimeData &timedata,
                    TimeIntegrator time_integrator,
                    libMesh::Order p_order,
                    const GetPot &data,
                    EquationType type = EquationType::PARABOLIC );

void assemble_forcing_and_eval_error(libMesh::EquationSystems &es, const TimeData &timedata, TimeIntegrator time_integrator, libMesh::Order p_order, const GetPot &data, IonicModel& ionic_model, std::vector<double>& error );


void ForcingTermConvergence(libMesh::EquationSystems &es, const double dt, const double Beta, const double Cm, const double SigmaSI, const double SigmaSE, const double SigmaBath, const double CurTime, const double xDim, const double v0, const double v1, const double v2, const double kcubic );

void init_cd_exact (EquationSystems & es, const double xDim);

void SolverRecovery (EquationSystems & es, const GetPot &data, TimeData& datatime);

double exact_solutionV_all (const double x,
              const double y,
              const int typePlace,
              const double time,
              const double z,
              const double xDim){

    double value = 0.;
    //double pi = acos(-1);
    double y0 = -.5;
    double c = .125;
    double sigma = .125;
    double delta = .1;
    double alpha = 50.;
    double a1 = 8.0/(xDim*xDim/4.0);
    double b1 = 16.0/(xDim/2.0);
    double c1 = 8.0;
    double a2 = -8.0/(xDim*xDim/4.0);
    double b2 = -8.0/(xDim/2.0);
    double c2 = -1.0;


    double dummy = y0 - y + c*time;

    double V = tanh(alpha*(dummy))*.5 + .5;
    double Vx = 0.;
    double Vxx = 0.;
    double Vy = .5*alpha*(pow(tanh(alpha*(dummy)),2) - 1.0);
    double Vyy = alpha*alpha*tanh(alpha*(dummy))*(pow(tanh(alpha*(dummy)),2) - 1.0);
    double Vxy = 0.0;
    double Vt = -(.5*alpha*c*(pow(tanh(alpha*dummy),2) - 1.0));

    double sigma6 = pow(sigma,6);
    double innerPow1 = delta - dummy;
    double innerPow1y = 1.0;
    double innerPow1t = -c;
    double innerPow2 = delta + dummy;
    double innerPow2y = -1.0;
    double innerPow2t = c;

    double exp1 = exp(-pow((innerPow1),6)/sigma6);
    double exp1y = -6.0*exp1*pow(innerPow1,5)*innerPow1y/sigma6;
    double exp1yy =  -30.0*exp1*pow(innerPow1,4)*innerPow1y*innerPow1y/sigma6 - 6.0*exp1y*pow(innerPow1,5)*innerPow1y/sigma6;
    double exp2 = exp(-pow((innerPow2),6)/sigma6);
    double exp2y = -6.0*exp2*pow(innerPow2,5)*innerPow2y/sigma6;
    double exp2yy =  -30.0*exp2*pow(innerPow2,4)*innerPow2y*innerPow2y/sigma6 - 6.0*exp2y*pow(innerPow2,5)*innerPow2y/sigma6;

    double f = exp1 - exp2;
    double fy = exp1y - exp2y;
    double fyy = exp1yy - exp2yy;
    if(x > 0){
      b1 = -b1;
      b2 = -b2;
    }

    double g1 = a1*x*x + b1*x + c1;
    double g1x = 2.0*a1*x + b1;
    double g1xx = 2.0*a1;
    double g2 = a2*x*x + b2*x + c2;
    double g2x = 2.0*a2*x + b2;
    double g2xx = 2.0*a2;

    double Ve = f;
    double Vex = 0.0;
    double Vexx = 0.0;
    double Vey = fy;
    double Veyy = fyy;
    double g, gx, gxx;

    if(x <= (-3.0/4.0)*xDim/2.0){
      g = g1;
      gx = g1x;
      gxx = g1xx;
    }
    else if(x > (-3.0/4.0)*xDim/2.0 && x < -xDim/4.0){
      g = g2;
      gx = g2x;
      gxx = g2xx;
    }
    else if(x >= (3.0/4.0)*xDim/2.0){
      g = g1;
      gx = g1x;
      gxx = g1xx;
    }
    else{
      g = g2;
      gx = g2x;
      gxx = g2xx;
    }


    double Vb = f*g;
    double Vbx = f*gx;
    double Vby = fy*g;
    double Vbxx = f*gxx;
    double Vbyy = fyy*g;
    double Vbxy = fy*gx;



    // Rest of tissue part: V
    if(typePlace == 2){
      //libMesh::out << .5 -.5*tanh(3.*pi*std::abs(.7)/(yDim/2.0) - pi - (1.0/5.0)*time)<< " " << std::abs(.7) << std::endl;
      //value = .5 -.5*tanh(3.*pi*(std::abs(y))/(yDim/2.0) - pi - (1.0/5.0)*time);
      value = V;
      //libMesh::out << pow(.5,2) << std::endl;
    }
    // Rest of tissue part: V time derivative
  else if(typePlace == 22){
    //value = (-1./10.0)*(1 - pow(tanh(-3.*pi*(std::abs(y))/(yDim/2.0) + pi + (1.0/5.0)*time),2));
    value = Vt;
  }
    // Rest of tissue part: V first derivative
  else if(typePlace == 20){
    double dummy2 = tanh(dummy);
    value = Vy;
  }
    // Rest of tissue part: V second derivative
  else if(typePlace == 200){
    //value = 9.*(pow(pi,2))*(1. - pow(tanh(3.*pi*(std::abs(y))/(yDim/2.0) - pi - (1.0/5.0)*time),2))*(tanh(3.*pi*(std::abs(y))/(yDim/2.0) - pi - (1.0/5.0)*time));
    value = Vyy;
  }



    // Rest of tissue part: Ve and Vb
    else if(typePlace == 3){
      //value = -.5 +.5*tanh(3.*pi*(std::abs(y))/(yDim/2.0) - pi - (1.0/5.0)*time);
      //double expo1 = pow(((-dummy+delta)/sigma),6);
      //double expo2 = pow(((dummy+delta)/sigma),6);
      //value = exp(expo1) - exp(expo2);
      value = Ve;

    }

    else if(typePlace == 4){
     value = Vb;
    }


    // Rest of tissue part: Ve and Vb first derivative
    else if(typePlace == 30){
      //value = (-.25*exp(-pow(((x - time*.15 + 2.0)/(2.0/8.0)),2)) + .25*exp(-pow(((x - time*.15 + 1.5)/(2.0/8.0)),2)))*(1.0 - abs(y)*2.0);
      //value = (3.*pi*y/(2.*(std::abs(y))))*(1. - pow(tanh(3.*pi*(std::abs(y))/(yDim/2.0) - pi - (1.0/5.0)*time),2));
      //double expo1 = pow(((dummy+delta)/sigma),5);
      //double expo2 = pow(((dummy-delta)/sigma),5);
      value = Vey;
    }
    else if(typePlace == 40){

    value = Vby;
  }

    // Rest of tissue part: Ve and Vb second derivative
    else if(typePlace == 300){
      //value = (-.25*exp(-pow(((x - time*.15 + 2.0)/(2.0/8.0)),2)) + .25*exp(-pow(((x - time*.15 + 1.5)/(2.0/8.0)),2)))*(1.0 - abs(y)*2.0);
      //value = -9.*(pow(pi,2))*(1. - pow(tanh(3.*pi*(std::abs(y))/(yDim/2.0) - pi - (1.0/5.0)*time),2))*(tanh(3.*pi*(std::abs(y))/(yDim/2.0) - pi - (1.0/5.0)*time));
      value = Veyy;
      //libMesh::out << value << "     place: " << typePlace  << " y: " << y << std::endl;
    }
    //summation of both the derivative in respect to y and x of Vb
    else if(typePlace == 4004){
      value = Vbyy + Vbxx;
    }




    //libMesh::out << value << "     place: " << typePlace<< std::endl;
    return value;
}


double CalculateF(const double x,
          const double y,
          const int typeEqn,
          const double time,
          const double z,
          const double dt1,
          const double Beta,
          const double Cm,
          const double SigmaSI,
          const double SigmaSE,
          const double SigmaBath,
          const double xDim,
          const double v0,
          const double v1,
          const double v2,
          const double kcubic){

  double value;

  //if(alphaTime != 1){

    //dt1 = alphaTime*(yDim/numEl)/(.125);
  //}


  //V equation, parabolic, so Fv
  if(typeEqn == 0){
    double prevTime = time - dt1;
    double prevprevTime = time - dt1*2.0;
    double u_current = exact_solutionV_all(x,y, 2, time, 0.0, xDim);
    double u_prev = exact_solutionV_all(x,y, 2, prevTime, 0.0, xDim);
    double u_prev_prev = exact_solutionV_all(x,y, 2, prevprevTime, 0.0, xDim);

    value = (1*Cm)*exact_solutionV_all(x,y, 22, time, 0.0, xDim) - 1.*( -kcubic*((u_current - v0)*(u_current - v1)*(u_current - v2)) ) - SigmaSI*exact_solutionV_all(x,y, 200, time, 0.0, xDim)/Beta - SigmaSI*exact_solutionV_all(x,y, 300, time, 0.0, xDim)/Beta;

  }
  //Ve equation, elliptic, so Fve
  else if(typeEqn == 1){
    value = -SigmaSI*exact_solutionV_all(x,y, 200, time, 0.0, xDim)/Beta - (SigmaSI + SigmaSE)*exact_solutionV_all(x,y, 300, time, 0.0, xDim)/Beta;
  }
  //Bath equation, Vb, so Fvb
  else if(typeEqn == 2){
    value = -SigmaBath*exact_solutionV_all(x,y, 4004, time, 0.0, xDim)/Beta;
  }

  return value;
}




double HofX(double uvalue, double otherValue){
  double Hvalue;

  if(uvalue - otherValue > 0){
    Hvalue = 1.;
  }
  else{
    Hvalue = 0.;
  }

  return Hvalue;
}





int main(int argc, char **argv)
{
    // Read input file
    GetPot cl(argc, argv);
    std::string datafile_name = cl.follow("data.beat", 2, "-i", "--input");
    GetPot data(datafile_name);
    // Use libMesh stuff without writing libMesh:: everytime
    using namespace libMesh;
    // Initialize libMesh
    LibMeshInit init(argc, argv, MPI_COMM_WORLD);

    // Create folde in which we save the output of the solution
    std::string output_folder = data("output_folder", "Output");
    std::string stimulus_type = data("stimulus_type", "Transmembrane");
    double IstimD = data("stimulus_duration",2.);

    struct stat out_dir;
    if (stat(&output_folder[0], &out_dir) != 0)
    {
        if (init.comm().rank() == 0)
        {
            mkdir(output_folder.c_str(), 0777);
        }
    }

    // Create empty mesh
    ParallelMesh mesh(init.comm());
    read_mesh(data, mesh);
    // output the details about the mesh
    mesh.print_info();

    // The dimension that we are running
    const unsigned int dim = mesh.mesh_dimension();

    // Time stuff
    TimeData datatime(data);
    datatime.print();

    // Pick time integrator
    // This will define if we use 1 or 2 systems
    std::map<std::string, TimeIntegrator> time_integrator_map =
    {
    { "SBDF1", TimeIntegrator::SBDF1 },
    { "SBDF2", TimeIntegrator::SBDF2 },
    { "SBDF3", TimeIntegrator::SBDF3 },
    { "EXPLICIT_EXTRACELLULAR", TimeIntegrator::EXPLICIT_EXTRACELLULAR },
    { "EXPLICIT_INTRACELLULAR", TimeIntegrator::EXPLICIT_INTRACELLULAR },
    { "SEMI_IMPLICIT", TimeIntegrator::SEMI_IMPLICIT },
    { "SEMI_IMPLICIT_HALF_STEP", TimeIntegrator::SEMI_IMPLICIT_HALF_STEP } };

    std::string integrator = data("integrator", "SBDF1");
    auto it = time_integrator_map.find(integrator);
    TimeIntegrator time_integrator = it->second;
    bool using_implicit_time_integrator = false;
    if (time_integrator == TimeIntegrator::SBDF1 || time_integrator == TimeIntegrator::SBDF2 || time_integrator == TimeIntegrator::SBDF3)
    {
        using_implicit_time_integrator = true;
    }
    bool convergence_test = data("convergence_test", false);


    // Create libMesh Equations systems
    // This will hold the mesh and create the corresponding
    // finite element spaces
    libMesh::EquationSystems es(mesh);

    // Define tissue active subdomain for subdomain restricted variables
    std::set < libMesh::subdomain_id_type > tissue_subdomains;
    tissue_subdomains.insert(static_cast<libMesh::subdomain_id_type>(Subdomain::TISSUE));
    // Define finite element ORDER
    int p_order = data("p_order", 1);
    std::cout << "P order: " << p_order << std::endl;
    Order order = FIRST;
    if (p_order == 2)
        order = SECOND;


    // Bidomain System
    switch (time_integrator)
    {
        case TimeIntegrator::EXPLICIT_EXTRACELLULAR:
        case TimeIntegrator::EXPLICIT_INTRACELLULAR:
        case TimeIntegrator::SEMI_IMPLICIT:
        case TimeIntegrator::SEMI_IMPLICIT_HALF_STEP:
        {
            libMesh::LinearImplicitSystem &elliptic_system = es.add_system < libMesh::LinearImplicitSystem > ("elliptic");
            libMesh::TransientLinearImplicitSystem & parabolic_system = es.add_system < libMesh::TransientLinearImplicitSystem > ("parabolic");
            TransientExplicitSystem & recovery = es.add_system <TransientExplicitSystem> ("Recovery");

            std::cout << "Using element of order p = " << order << std::endl;

            parabolic_system.add_variable("V", order, LAGRANGE, &tissue_subdomains);
            elliptic_system.add_variable("Ve", order, LAGRANGE);
            parabolic_system.add_matrix("Ki");
            parabolic_system.add_matrix("Ke");
            parabolic_system.add_matrix("Kg");
            parabolic_system.add_matrix("M");
            parabolic_system.add_vector("ML");
            parabolic_system.add_vector("In");
            parabolic_system.add_vector("FV"); // parabolic forcing term for exact solution
            parabolic_system.add_vector("aux1"); // auxilliary vector for assembling the RHS
            elliptic_system.add_vector("FVe"); // elliptic forcing term for exact solution
            elliptic_system.add_vector("I_extra_stim"); // elliptic forcing term for exact solution

            recovery.add_variable("v", order, LAGRANGE, &tissue_subdomains);
            recovery.add_variable("w", order, LAGRANGE, &tissue_subdomains);
            recovery.add_variable("s", order, LAGRANGE, &tissue_subdomains);
            

            es.init();
            if(convergence_test)
            {
                libMesh::AnalyticFunction<libMesh::Number> ic_V(exact_solution_V);
                libMesh::AnalyticFunction<libMesh::Number> ic_Ve_Vb(exact_solution_Ve_Vb);
                parabolic_system.project_solution(&ic_V);
                elliptic_system.project_solution(&ic_Ve_Vb);
            }

            break;
        }
        case TimeIntegrator::SBDF1:
        case TimeIntegrator::SBDF2:
        case TimeIntegrator::SBDF3:
        default:
        {
            libMesh::TransientLinearImplicitSystem &system = es.add_system < libMesh::TransientLinearImplicitSystem > ("bidomain");
            std::cout << "Using element of order p = " << order << std::endl;

            
            TransientExplicitSystem & recovery = es.add_system <TransientExplicitSystem> ("Recovery");

            if(convergence_test){
                LinearImplicitSystem & solutionSystem = es.add_system <LinearImplicitSystem> ("Solution");
                solutionSystem.add_variable("Ve_exact", order, LAGRANGE);
                solutionSystem.add_variable("V_exact",order, LAGRANGE, &tissue_subdomains);
            } 

            system.add_variable("V", order, LAGRANGE, &tissue_subdomains);
            system.add_variable("Ve", order, LAGRANGE);
            system.add_matrix("K");
            system.add_matrix("M");
            system.add_vector("ML");
            system.add_vector("In");
            system.add_vector("Inm1");
            system.add_vector("Inm2");
            system.add_vector("Vnm1");
            system.add_vector("Vnm2");
            system.add_vector("F"); // forcing term for exact solution
            system.add_vector("aux1"); // auxilliary vector for assembling the RHS
            system.add_vector("ForcingConv");
            system.add_vector("I_extra_stim");

            recovery.add_variable("v", order, LAGRANGE, &tissue_subdomains);
            recovery.add_variable("w", order, LAGRANGE, &tissue_subdomains);
            recovery.add_variable("s", order, LAGRANGE, &tissue_subdomains);
            recovery.add_vector("v_prev");
            recovery.add_vector("w_prev");
            recovery.add_vector("s_prev");
            recovery.add_vector("v_prev_prev");
            recovery.add_vector("w_prev_prev");
            recovery.add_vector("s_prev_prev");   

            es.init();

            if(convergence_test)
            {
                //libMesh::AnalyticFunction<libMesh::Number> ic(exact_solution_monolithic);
                //system.project_solution(&ic);
            }

            break;
        }
    }

    es.print_info();

    // setup pacing protocol
    Pacing pacing(data);
    pacing.print();
    IonicModel ionic_model;
    ionic_model.setup(data);
    ionic_model.print();



    int save_iter = 0;
    {
        std::cout << "Time: " << datatime.time << ", " << std::flush;
        std::cout << "Exporting  ... " << std::flush;
        std::ostringstream ss;
        ss << std::setw(4) << std::setfill('0') << save_iter;
        std::string step_str = ss.str();
        libMesh::VTKIO(mesh).write_equation_systems(output_folder+"/bath_" + step_str + ".pvtu", es);
        std::cout << "done " << std::endl;
    }

    // Start loop in time

    double totalErrorInSpaceTimeV, totalErrorInSpaceTimeVe, totalErrorInSpaceTimeVb;
    double xDim = data("maxx",1.) - data("minx", -1.);

    std::vector<double> error(3);

    for (; datatime.time < datatime.end_time; )
    {
        if(convergence_test)
        {
            //assemble_forcing_and_eval_error(es, datatime, time_integrator, order, data, ionic_model, error);
        }
        // advance time
        datatime.time += datatime.dt;
        datatime.timestep++;
        //std::cout << "Time: " << datatime.time << std::endl;
        // advance vectors and solve
        // Bidomain System
        switch (time_integrator)
        {
            case TimeIntegrator::EXPLICIT_EXTRACELLULAR:
            case TimeIntegrator::EXPLICIT_INTRACELLULAR:
            case TimeIntegrator::SEMI_IMPLICIT:
            case TimeIntegrator::SEMI_IMPLICIT_HALF_STEP:
            {
                if(datatime.timestep == 1)
                {
                    assemble_matrices(es, datatime, time_integrator, order, data);
                }
                libMesh::LinearImplicitSystem &elliptic_system = es.get_system < libMesh::LinearImplicitSystem > ("elliptic");
                libMesh::TransientLinearImplicitSystem & parabolic_system = es.get_system < libMesh::TransientLinearImplicitSystem > ("parabolic");

                *parabolic_system.old_local_solution = *parabolic_system.solution;
                //solve_ionic_model_evaluate_ionic_currents(es, ionic_model, pacing, datatime, time_integrator);
                SolverRecovery (es, data, datatime);
                // Solve Parabolic
                assemble_rhs(es, datatime, time_integrator, order, data, EquationType::PARABOLIC);
                if(time_integrator == TimeIntegrator::SEMI_IMPLICIT ||
                   time_integrator == TimeIntegrator::SEMI_IMPLICIT_HALF_STEP)
                {
                    parabolic_system.solve();
                }
                else
                {
                    *parabolic_system.solution += *parabolic_system.rhs;
                }
                parabolic_system.update();
                // Solve Elliptic
                assemble_rhs(es, datatime, time_integrator, order, data, EquationType::ELLIPTIC);

                if(stimulus_type.compare(0,13,"Extracellular") == 0){
                    if(datatime.timestep*datatime.dt < IstimD){
                        *elliptic_system.rhs += elliptic_system.get_vector("I_extra_stim");
                        libMesh::out << "Extracellular stimulus happening..." << std::endl;
                    }
                }

                elliptic_system.solve();
                elliptic_system.update();

                if(time_integrator == TimeIntegrator::SEMI_IMPLICIT_HALF_STEP)
                {
                     // Solve Parabolic
                     assemble_rhs(es, datatime, time_integrator, order, data, EquationType::PARABOLIC);
                     parabolic_system.solve();
                     parabolic_system.update();
                }
                break;
            }
            case TimeIntegrator::SBDF3:
            {
                if(datatime.timestep == 1)
                {
                    assemble_matrices(es, datatime, TimeIntegrator::SBDF1, order, data);
                    if(convergence_test){
                        init_cd_exact(es, xDim);
                    }
                }
                if(datatime.timestep == 2)
                {
                    assemble_matrices(es, datatime, TimeIntegrator::SBDF2, order, data);
                }
                if(datatime.timestep == 3)
                {
                    assemble_matrices(es, datatime, time_integrator, order, data);
                }

                libMesh::TransientLinearImplicitSystem &system = es.get_system < libMesh::TransientLinearImplicitSystem > ("bidomain");

                system.get_vector("Inm2") = system.get_vector("Inm1");
                system.get_vector("Inm1") = system.get_vector("In");
                system.get_vector("Vnm2") = *system.older_local_solution;
                system.get_vector("Vnm1") = *system.old_local_solution;
                *system.older_local_solution = *system.old_local_solution;
                *system.old_local_solution = *system.solution;
                //solve_ionic_model_evaluate_ionic_currents(es, ionic_model, pacing, datatime, time_integrator);
                SolverRecovery (es, data, datatime);
                if(datatime.timestep == 1) assemble_rhs(es, datatime, TimeIntegrator::SBDF1, order, data);
                else if(datatime.timestep == 2) assemble_rhs(es, datatime, TimeIntegrator::SBDF2, order, data);
                else assemble_rhs(es, datatime, time_integrator, order, data);

                if(stimulus_type.compare(0,13,"Extracellular") == 0){
                    if(datatime.timestep*datatime.dt < IstimD){
                        *system.rhs += system.get_vector("I_extra_stim");
                        libMesh::out << "Extracellular stimulus happening..." << std::endl;
                    }
                }

                system.solve();
                system.update();
                break;
            }
            case TimeIntegrator::SBDF2:
            {
                if(datatime.timestep == 1)
                {
                    assemble_matrices(es, datatime, TimeIntegrator::SBDF1, order, data);
                    if(convergence_test){
                        init_cd_exact(es, xDim);
                    }
                }
                if(datatime.timestep == 2)
                {
                    assemble_matrices(es, datatime, time_integrator, order, data);
                }

                libMesh::TransientLinearImplicitSystem &system = es.get_system < libMesh::TransientLinearImplicitSystem > ("bidomain");

                system.get_vector("Inm2") = system.get_vector("Inm1");
                system.get_vector("Inm1") = system.get_vector("In");
                system.get_vector("Vnm2") = *system.older_local_solution;
                system.get_vector("Vnm1") = *system.old_local_solution;
                *system.older_local_solution = *system.old_local_solution;
                *system.old_local_solution = *system.solution;
                //solve_ionic_model_evaluate_ionic_currents(es, ionic_model, pacing, datatime, time_integrator);
                SolverRecovery (es, data, datatime);
                if(datatime.timestep == 1) assemble_rhs(es, datatime, TimeIntegrator::SBDF1, order, data);
                else assemble_rhs(es, datatime, time_integrator, order, data);
                //system.rhs->print();
                //system.get_vector("ML").print();
                //system.get_matrix("M").print();
                //system.matrix->print();

                if(stimulus_type.compare(0,13,"Extracellular") == 0){
                    if(datatime.timestep*datatime.dt < IstimD){
                        *system.rhs += system.get_vector("I_extra_stim");
                        libMesh::out << "Extracellular stimulus happening..." << std::endl;
                    }
                }

                system.solve();
                //system.solution->print();
                system.update();
                break;
            }
            case TimeIntegrator::SBDF1:
            default:
            {
                if(datatime.timestep == 1){
                    assemble_matrices(es, datatime, time_integrator, order, data);
                    if(convergence_test){
                        init_cd_exact(es, xDim);
                    }
                }

                libMesh::TransientLinearImplicitSystem &system = es.get_system < libMesh::TransientLinearImplicitSystem > ("bidomain");
                system.get_vector("Inm2") = system.get_vector("Inm1");
                system.get_vector("Inm1") = system.get_vector("In");
                system.get_vector("Vnm2") = *system.older_local_solution;
                system.get_vector("Vnm1") = *system.old_local_solution;
                *system.older_local_solution = *system.old_local_solution;
                *system.old_local_solution = *system.solution;
                //solve_ionic_model_evaluate_ionic_currents(es, ionic_model, pacing, datatime, time_integrator);
                SolverRecovery (es, data, datatime);
                assemble_rhs(es, datatime, time_integrator, order, data);

                if(stimulus_type.compare(0,13,"Extracellular") == 0){
                    if(datatime.timestep*datatime.dt < IstimD){
                        *system.rhs += system.get_vector("I_extra_stim");
                        libMesh::out << "Extracellular stimulus happening..." << std::endl;
                    }
                }

                system.solve();
                system.update();
                break;
            }
        }

        //Export the solution if at the right timestep
        if (0 == datatime.timestep % datatime.export_timesteps)
        {
            std::cout << "Time: " << datatime.time << ", " << std::flush;
            std::cout << "Exporting  ... " << std::flush;
            // update output file counter
            save_iter++;//
            std::ostringstream ss;
            ss << std::setw(4) << std::setfill('0') << save_iter;
            std::string step_str = ss.str();
            libMesh::VTKIO(mesh).write_equation_systems(output_folder+"/bath_" + step_str + ".pvtu", es);
            std::cout << "done " << std::endl;
        }




     //START OF ERROR CALCULATION
      if(convergence_test && (time_integrator == TimeIntegrator::SBDF1 || time_integrator == TimeIntegrator::SBDF2 || time_integrator == TimeIntegrator::SBDF3) ){

          double totalErrorInSpaceV = 0.;
          double totalErrorInSpaceVe = 0.;
          double totalErrorInSpaceVb = 0.;

          libMesh::LinearImplicitSystem &solutionSystem = es.get_system < libMesh::LinearImplicitSystem > ("Solution");

          auto femSolu = es.get_system("bidomain").variable_number("V");
          auto femSolu2 = es.get_system("bidomain").variable_number("Ve");
          auto femSoluV_exact = es.get_system("Solution").variable_number("V_exact");
          auto femSoluVe_exact = es.get_system("Solution").variable_number("Ve_exact");


          const unsigned int dim = mesh.mesh_dimension();

          const DofMap & dof_map = es.get_system("Solution").get_dof_map();
          const DofMap & dof_map2 = es.get_system("Recovery").get_dof_map();

          FEType fe_type = dof_map.variable_type(femSolu);
          std::unique_ptr<FEBase> fe (FEBase::build(dim, fe_type));

          QGauss qrule (dim, TENTH);

          // Tell the finite element object to use our quadrature rule.
          fe->attach_quadrature_rule (&qrule);

          const std::vector<Point> & q_point = fe->get_xyz();
          const std::vector<Real> & JxW = fe->get_JxW();
          const std::vector<std::vector<Real>> & phi = fe->get_phi();

          int rowsn = 0;
          int colsn = 0;

          double u_h, u_exact, ue_h, ue_exact, ub_h, ub_exact;

          std::vector<dof_id_type> dof_indices;
          std::vector<dof_id_type> dof_indices2;


          for(const auto & node : mesh.local_node_ptr_range()){

            dof_map.dof_indices (node, dof_indices);
            dof_map2.dof_indices (node, dof_indices2);

            const Real x = (*node)(0);
            const Real y = (*node)(1);

            if(dof_indices2.size() > 0){

            u_exact = exact_solutionV_all(x, y, 2, datatime.time, 0.0, xDim);
            solutionSystem.solution -> set(dof_indices[femSoluV_exact],u_exact);
            ue_exact = exact_solutionV_all(x, y, 3, datatime.time, 0.0, xDim);
            solutionSystem.solution -> set(dof_indices[femSoluVe_exact],ue_exact);

            }
            else{
            ue_exact = exact_solutionV_all(x, y, 4, datatime.time, 0.0, xDim);
            solutionSystem.solution -> set(dof_indices[femSoluVe_exact],ue_exact);
            }



            //libMesh::out << "x: " << x << "      dof: " << dof_indices[femSoluVe_exact] << "    ue= " << ue_exact << std::endl;
            //totalErrorInSpaceV += 1*((u_h - u_exact)*(u_h - u_exact));
            //totalErrorInSpaceVe += 1*((ue_h - ue_exact)*(ue_h - ue_exact));

          }







          for(const auto & elem : mesh.active_local_element_ptr_range()){

          fe->reinit (elem);

          for (unsigned int qp=0; qp<qrule.n_points(); qp++){

            if(elem -> subdomain_id() == 1){
              u_h = es.get_system("bidomain").point_value(femSolu, q_point[qp], elem);
              u_exact = exact_solutionV_all(q_point[qp](0), q_point[qp](1), 2, datatime.time, 0.0, xDim);
              ue_h = es.get_system("bidomain").point_value(femSolu2, q_point[qp], elem);
              ue_exact = exact_solutionV_all(q_point[qp](0),q_point[qp](1), 3, datatime.time, 0.0, xDim);
              ub_h = 0.;
              ub_exact = 0.;

              //equation_systems.get_system("Bidomain").point_value(femSoluV_exact, q_point[qp], elem) = u_exact;
              //equation_systems.get_system("Bidomain").point_value(femSoluVe_exact, q_point[qp], elem) = ue_exact;

            }
            else{
              u_h = 0.;
              u_exact = 0.;
              ue_h = 0.;
              ue_exact = 0.;
              ub_h = es.get_system("bidomain").point_value(femSolu2, q_point[qp], elem);
              ub_exact = exact_solutionV_all(q_point[qp](0),q_point[qp](1), 4, datatime.time, 0.0, xDim);

              //equation_systems.get_system("Bidomain").point_value(femSoluVe_exact, q_point[qp], elem) = ue_exact;
            }


            totalErrorInSpaceV += JxW[qp]*((u_h - u_exact)*(u_h - u_exact));
            totalErrorInSpaceVe += JxW[qp]*((ue_h - ue_exact)*(ue_h - ue_exact));
            totalErrorInSpaceVb += JxW[qp]*((ub_h - ub_exact)*(ub_h - ub_exact));
            //rowsn = double(elem);
            //colsn = qp;
            //cout << qp << '\n';

            //if(qp == 0){
              //rowsn = rowsn + 1;
              //cout << 'this is rows: ' << rowsn << '\n';
            //}

            //cout << 'element: ' << elem << ' with qp: ' << q_point[qp] << endl;
            //cout << typeof(elem).name() << endl;
          }

      }


      //cout << rowsn << " and " << colsn+1 << '\n';

      totalErrorInSpaceTimeV += datatime.dt*std::sqrt(totalErrorInSpaceV);
      totalErrorInSpaceTimeVe += datatime.dt*std::sqrt(totalErrorInSpaceVe);
      totalErrorInSpaceTimeVb += datatime.dt*std::sqrt(totalErrorInSpaceVb);

      libMesh::out << "The L2 norm for V in space is: " << std::sqrt(totalErrorInSpaceV)<< std::endl;
      libMesh::out << "The L2 norm for Ve in space is: " << std::sqrt(totalErrorInSpaceVe)<< std::endl;
      libMesh::out << "The L2 norm for Vb in space is: " << std::sqrt(totalErrorInSpaceVb)<< std::endl;

    }
    //END OF ERROR CALCULATION





    }


    if(convergence_test)
    {
        //assemble_forcing_and_eval_error(es, datatime, time_integrator, order, data, ionic_model, error);
        //std::cout << "Error V: " << error[0] << std::endl;
        //std::cout << "Error Ve: " << error[1] << std::endl;
        //std::cout << "Error Vb: " << error[2] << std::endl;






        libMesh::out << "The L2 norm for V in space AND time is: " << (totalErrorInSpaceTimeV)<< std::endl;
        libMesh::out << "The L2 norm for Ve in space AND time is: " << (totalErrorInSpaceTimeVe)<< std::endl;
        libMesh::out << "The L2 norm for Vb in space AND time is: " << (totalErrorInSpaceTimeVb)<< std::endl;

        ofstream outputFile;
        std::string nel2 = data("nelx", "50");
        std::string eltype2 = data("eltype", "QUAD4");

        std::string eltype = data("eltype", "simplex");
        std::string nameOfOutputFile = "outPut_for_"+integrator+"_"+nel2+"elems_"+eltype2+"type.txt";
        outputFile.open (nameOfOutputFile);
        outputFile << "The L2 norm for V in space AND time is: " << (totalErrorInSpaceTimeV)<< std::endl;
        outputFile << "The L2 norm for Ve in space AND time is: " << (totalErrorInSpaceTimeVe)<< std::endl;
        outputFile << "The L2 norm for Vb in space AND time is: " << (totalErrorInSpaceTimeVb)<< std::endl;
        //outputFile << "Operation took: "<<time_span.count()<<" seconds"<<std::endl;
        outputFile.close();



    }

    return 0;
}

void read_mesh(const GetPot &data, libMesh::ParallelMesh &mesh)
{
    using namespace libMesh;
    // read mesh from file?
    std::string mesh_filename = data("mesh_filename", "NONE");
    if (mesh_filename.compare("NONE") != 0)
    {
        // READ MESH
        std::cout << "I will read the mesh here: " << mesh_filename << std::endl;
    }
    else
    {
        // Create Mesh
        // number of elements in the x,y and z direction
        int nelx = data("nelx", 10);
        int nely = data("nely", 10);
        // if nelz = 0 we run in 2D
        int nelz = data("nelz", 0);

        // the cube dimensions are defined as [minx, maxx] x [miny, maxy] x [minz, maxz]
        double maxx = data("maxx", 1.0);
        double maxy = data("maxy", 1.0);
        double maxz = data("maxz", 1.0);

        double minx = data("minx", 0.0);
        double miny = data("miny", 0.0);
        double minz = data("minz", 0.0);

        // Get mesh parameters
        std::string eltype = data("eltype", "simplex");
        std::map < std::string, ElemType > elem_type_map =
        {
        { "TRI3", TRI3 },
        { "TRI6", TRI6 },
        { "QUAD4", QUAD4 },
        { "QUAD9", QUAD9 },
        { "TET4", TET4 },
        { "HEX8", HEX8 } };
        auto elType = TRI3;
        auto elem_type_it = elem_type_map.find(eltype);
        if (elem_type_it != elem_type_map.end())
            elType = elem_type_it->second;

        std::cout << "Creating the cube [" << minx << ", " << maxx << "] x [" << miny << ", " << maxy << "] x [" << minx << ", " << maxx << "] " << std::endl;
        std::cout << "Using " << nelx << " x " << nely << " x " << nelz << " elements." << std::endl;
        std::cout << "Element type " << elem_type_it->first << std::endl;

        // Create mesh
        MeshTools::Generation::build_cube(mesh, nelx, nely, nelz, minx, maxx, miny, maxy, minz, maxz, elType);

        // setup subdomains
        // tissue domain box
        double tissue_maxx = data("tissue_maxx", 1.0);
        double tissue_maxy = data("tissue_maxy", 1.0);
        double tissue_maxz = data("tissue_maxz", 1.0);
        double tissue_minx = data("tissue_minx", 0.0);
        double tissue_miny = data("tissue_miny", 0.0);
        double tissue_minz = data("tissue_minz", 0.0);
        libMesh::BoundingBox box(libMesh::Point(tissue_minx, tissue_miny, tissue_minz), libMesh::Point(tissue_maxx, tissue_maxy, tissue_maxz));

        // right_interface
        for (auto el : mesh.element_ptr_range())
        {
            auto c = el->centroid();
            // Are we in the tissue region?
            if (box.contains_point(c))
            {
                el->subdomain_id() = static_cast<libMesh::subdomain_id_type>(Subdomain::TISSUE);
            }
            // we are not
            else
            {
                el->subdomain_id() = static_cast<libMesh::subdomain_id_type>(Subdomain::BATH);
            }
        }
    }
}

// Initialize TimeData from input
TimeData::TimeData(GetPot &data) :
        time(0.0),
        end_time(data("end_time", 1.0)),
        dt(data("dt", 0.125)),
        timestep(0),
        export_timesteps(data("export_timesteps", 1))
{
}
// Initialize TimeData from input
void TimeData::print()
{
    std::cout << "**************************************** " << std::endl;
    std::cout << "TimeData Parameters: " << std::endl;
    std::cout << "End time: " << end_time << std::endl;
    std::cout << "Dt: " << dt << std::endl;
    std::cout << "Current Timestep: " << timestep << std::endl;
    std::cout << "Export Timesteps: " << export_timesteps << std::endl;
    std::cout << "**************************************** " << std::endl;
}

void assemble_matrices(libMesh::EquationSystems &es, const TimeData &timedata, TimeIntegrator time_integrator, libMesh::Order p_order, const GetPot &data)
{
    using namespace libMesh;
    // Create vector of BC sidesets ids
    std::set<int> bc_sidesets;
    std::string bcs = data("bcs", "");
    read_bc_list(bcs, bc_sidesets);


    const MeshBase &mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();

    // volume element
    FEType fe_type(p_order);
    std::unique_ptr < FEBase > fe(FEBase::build(dim, fe_type));
    Order qp_order = THIRD;
    if (p_order > 1)
        qp_order = FIFTH;
    libMesh::QGauss qrule(dim, qp_order);
    fe->attach_quadrature_rule(&qrule);
    // surface element
    std::unique_ptr < FEBase > fe_face(FEBase::build(dim, fe_type));
    QGauss qface(dim - 1, qp_order);
    fe_face->attach_quadrature_rule(&qface);

    // quantities for volume integration
    const std::vector<Real> &JxW = fe->get_JxW();
    const std::vector<Point> &q_point = fe->get_xyz();
    const std::vector<std::vector<Real>> &phi = fe->get_phi();
    const std::vector<std::vector<RealGradient>> &dphi = fe->get_dphi();

    // quantities for surface integration
    const std::vector<Real> &JxW_face = fe_face->get_JxW();
    const std::vector<Point> &q_point_face = fe_face->get_xyz();
    const std::vector<std::vector<Real>> &phi_face = fe_face->get_phi();
    const std::vector<std::vector<RealGradient>> &dphi_face = fe_face->get_dphi();
    const std::vector<Point> &normal = fe_face->get_normals();

    // define fiber field
    double fx = data("fx", 1.0), fy = data("fy", 0.0), fz = data("fz", 0.0);
    double sx = data("sx", 0.0), sy = data("sy", 1.0), sz = data("sz", 0.0);
    double nx = data("nx", 0.0), ny = data("ny", 0.0), nz = data("nz", 1.0);
    VectorValue<Number>  f0(fx, fy, fz);
    VectorValue<Number>  s0(sx, sy, sz);
    VectorValue<Number>  n0(nx, ny, nz);
    // setup conductivities:
    // Default parameters from
    // Cardiac excitation mechanisms, wavefront dynamics and strength�interval
    // curves predicted by 3D orthotropic bidomain simulations

    // read parameters
    double sigma_f_i = data("sigma_f_i", 2.3172);
    double sigma_s_i = data("sigma_s_i", 0.2435); // sigma_t in the paper
    double sigma_n_i = data("sigma_n_i", 0.0569);
    double sigma_f_e = data("sigma_f_e", 1.5448);
    double sigma_s_e = data("sigma_s_e", 1.0438);  // sigma_t in the paper
    double sigma_n_e = data("sigma_n_e", 0.37222);
    double sigma_b_ie = data("sigma_b", 6.0);
    double chi = data("chi", 1e3);
    double Cm = data("Cm", 1.5);
    double penalty = data("penalty", 1e8);
    double interface_penalty = data("interface_penalty", 1e4);
    double zDim = data("nelz",0);
    double stimulus_maxx = data("stimulus_maxx", .5);
    double stimulus_minx = data("stimulus_minx", -.5);
    double stimulus_maxy = data("stimulus_maxy", 1.5);
    double stimulus_miny = data("stimulus_miny", 1.3);
    double stimulus_maxz = data("stimulus_maxz", .05);
    double stimulus_minz = data("stimulus_minz", -.05);
    double IstimV = data("stimulus_amplitude", -1.);

    // setup tensors parameters
    // f0 \otimes f0
    TensorValue<Number> fof(fx * fx, fx * fy, fx * fz, fy * fx, fy * fy, fy * fz, fz * fx, fz * fy, fz * fz);
    // s0 \otimes s0
    TensorValue<Number> sos(sx * sx, sx * sy, sx * sz, sy * sx, sy * sy, sy * sz, sz * sx, sz * sy, sz * sz);
    // n0 \otimes n0
    TensorValue<Number> non(nx * nx, nx * ny, nx * nz, ny * nx, ny * ny, ny * nz, nz * nx, nz * ny, nz * nz);

    TensorValue<Number> sigma_i = ( sigma_f_i * fof + sigma_s_i * sos + sigma_n_i * non ) /chi;
    TensorValue<Number> sigma_e = ( sigma_f_e * fof + sigma_s_e * sos + sigma_n_e * non) / chi;
    TensorValue<Number> sigma_b(sigma_b_ie/chi, 0.0, 0.0, 0.0, sigma_b_ie/chi, 0.0, 0.0, 0.0, sigma_b_ie/chi);

    DenseMatrix < Number > K;
    DenseMatrix < Number > M;
    DenseVector < Number > M_lumped;
    DenseVector < Number > I_extra_stim;

    std::vector < dof_id_type > dof_indices;
    std::vector < dof_id_type > parabolic_dof_indices;
    std::vector < dof_id_type > elliptic_dof_indices;

    // Assemble matrices differently based on time integrator
    switch (time_integrator)
    {
        case TimeIntegrator::EXPLICIT_EXTRACELLULAR:
        case TimeIntegrator::EXPLICIT_INTRACELLULAR:
        case TimeIntegrator::SEMI_IMPLICIT:
        case TimeIntegrator::SEMI_IMPLICIT_HALF_STEP:
        {
            std::cout << "Assembling matrices for EXPLICIT EXTRACELLULAR: " << static_cast<int>(time_integrator) << std::endl;
            libMesh::LinearImplicitSystem &elliptic_system = es.get_system < libMesh::LinearImplicitSystem > ("elliptic");
            libMesh::TransientLinearImplicitSystem & parabolic_system = es.get_system < libMesh::TransientLinearImplicitSystem > ("parabolic");
            elliptic_system.zero_out_matrix_and_rhs = false;
            elliptic_system.assemble_before_solve = false;
            parabolic_system.zero_out_matrix_and_rhs = false;
            parabolic_system.assemble_before_solve = false;
            elliptic_system.matrix->zero();
            parabolic_system.matrix->zero();
            parabolic_system.get_matrix("Ki").zero();
            parabolic_system.get_matrix("Ke").zero();
            parabolic_system.get_matrix("Kg").zero();
            parabolic_system.get_matrix("M").zero();
            parabolic_system.get_vector("ML").zero();
            DenseMatrix < Number > Ki;
            DenseMatrix < Number > Ke;
            DenseMatrix < Number > Kg;
            DenseMatrix < Number > Ke_interface;
            DenseMatrix < Number > S;

            const DofMap & elliptic_dof_map = elliptic_system.get_dof_map();
            const DofMap & parabolic_dof_map = parabolic_system.get_dof_map();


            for (auto elem : mesh.element_ptr_range())
            {
                parabolic_dof_map.dof_indices(elem, parabolic_dof_indices);
                elliptic_dof_map.dof_indices(elem, elliptic_dof_indices);
                fe->reinit(elem);
                int elliptic_ndofs = elliptic_dof_indices.size();
                int parabolic_ndofs = parabolic_dof_indices.size();

                // resize local elemental matrices
                K.resize(elliptic_ndofs, elliptic_ndofs);
                Ki.resize(parabolic_ndofs, parabolic_ndofs);
                Ke.resize(parabolic_ndofs, parabolic_ndofs);
                Kg.resize(parabolic_ndofs, parabolic_ndofs);
                M.resize(parabolic_ndofs, parabolic_ndofs);
                S.resize(parabolic_ndofs, parabolic_ndofs);
                I_extra_stim.resize (elliptic_dof_indices.size());
                M_lumped.resize(parabolic_ndofs);

                auto subdomain_id = elem->subdomain_id();

                //std::cout << "Loop over volume: " << std::flush;

                // if we are in the bath
                if (subdomain_id == static_cast<short unsigned int>(Subdomain::BATH))
                {
                    for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
                    {
                        const Real x = q_point[qp](0);
                        const Real y = q_point[qp](1);
                        const Real z = q_point[qp](2);

                        for (unsigned int i = 0; i != elliptic_ndofs; i++)
                        {
                            for (unsigned int j = 0; j != elliptic_ndofs; j++)
                            {
                                K(i, j) += JxW[qp] * dphi[i][qp] * (sigma_b * dphi[j][qp]);



                                if(zDim == 0){
                                    if( y < stimulus_maxy && y > stimulus_miny && x > stimulus_minx && x < stimulus_maxx ){
                                        //libMesh::out << "top boundary ->    x: " << x << ";      y: " << y << std::endl;
                                        I_extra_stim(i) += JxW[qp]*(     phi[i][qp]*phi[j][qp]*((1./1.)*IstimV)     );
                                    }
                                    else{
                                        I_extra_stim(i) += JxW[qp]*(0.);
                                    }
                                }
                                else{
                                    if( y < stimulus_maxy && y > stimulus_miny && x > stimulus_minx && x < stimulus_maxx && z > stimulus_minz && z < stimulus_maxz ){
                                        //libMesh::out << "top boundary ->    x: " << x << ";      y: " << y << std::endl;
                                        I_extra_stim(i) += JxW[qp]*(     phi[i][qp]*phi[j][qp]*((1./1.)*IstimV)     );
                                    }
                                    else{
                                        I_extra_stim(i) += JxW[qp]*(0.);
                                    }
                                }


                            }
                        }
                    }
                }
                else
                {
                    for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
                    {
                        for (unsigned int i = 0; i != elliptic_ndofs; i++)
                        {
                            for (unsigned int j = 0; j != elliptic_ndofs; j++)
                            {
                                // Elliptic equation matrix
                                K(i, j) += JxW[qp] * dphi[i][qp] * ((sigma_i + sigma_e) * dphi[j][qp]);

                                // Parabolic matrices
                                // add mass matrix
                                M_lumped(i) += JxW[qp] * Cm / timedata.dt * phi[i][qp] * phi[j][qp];
                                // stiffness matrix
                                Ki(i, j) += JxW[qp] * dphi[i][qp] * ((sigma_i) * dphi[j][qp]);
                                Ke(i, j) += JxW[qp] * dphi[i][qp] * ((sigma_e) * dphi[j][qp]);
                                M(i,  j) += JxW[qp] * phi[i][qp] * phi[j][qp];
                                S(i, j)  += JxW[qp] * dphi[i][qp] * ((sigma_i) * dphi[j][qp]);
                                S(i, i)  += JxW[qp] * Cm / timedata.dt * phi[i][qp] * phi[j][qp];
                            }
                        }
                    }
                }
                //std::cout << ", Loop over interface: " << std::flush;

                // interface
                if( time_integrator == TimeIntegrator::EXPLICIT_EXTRACELLULAR )
                {
                    if (subdomain_id == static_cast<libMesh::subdomain_id_type>(Subdomain::TISSUE))
                    {
                        for (auto side : elem->side_index_range())
                        {
                            if (elem->neighbor_ptr(side) != nullptr)
                            {
                                auto neighbor_subdomain_id = elem->neighbor_ptr(side)->subdomain_id();
                                if(subdomain_id != neighbor_subdomain_id)
                                {
                                    //std::cout << "Assembling interface" << std::endl;
                                    fe_face->reinit(elem, side);
                                    for (unsigned int qp = 0; qp < qface.n_points(); qp++)
                                    {
                                        for (unsigned int i = 0; i != parabolic_ndofs; i++)
                                        {
                                            for (unsigned int j = 0; j != parabolic_ndofs; j++)
                                            {
                                                Ke(i, j) -= JxW_face[qp] * ( ( sigma_e * dphi_face[j][qp] ) * normal[qp] ) * phi_face[i][qp];
                                                Kg(i, j) -= interface_penalty * JxW_face[qp] * ( ( sigma_i * dphi_face[j][qp] ) * normal[qp] ) * phi_face[i][qp];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // boundaries
                //std::cout << ", Loop over boundaries: " << std::flush;
                if (subdomain_id == static_cast<libMesh::subdomain_id_type>(Subdomain::BATH))
                {
                    for (auto side : elem->side_index_range())
                    {
                        if (elem->neighbor_ptr(side) == nullptr)
                        {
                            auto boundary_id = mesh.get_boundary_info().boundary_id(elem, side);
                            auto it = bc_sidesets.find(static_cast<int>(boundary_id));
                            if (it != bc_sidesets.end())
                            {
                                //std::cout << "found_boundary " << std::endl;
                                fe_face->reinit(elem, side);
                                for (unsigned int qp = 0; qp < qface.n_points(); qp++)
                                {
                                    for (unsigned int i = 0; i != elliptic_ndofs; i++)
                                    {
                                        for (unsigned int j = 0; j != elliptic_ndofs; j++)
                                        {
                                            K(i, j) += JxW_face[qp] * penalty * phi_face[j][qp] * phi_face[i][qp];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                //std::cout << "Add matrix " << std::endl;

                elliptic_system.matrix->add_matrix (K, elliptic_dof_indices, elliptic_dof_indices);
                parabolic_system.matrix->add_matrix(S, parabolic_dof_indices, parabolic_dof_indices);
                parabolic_system.get_matrix("M").add_matrix(M, parabolic_dof_indices, parabolic_dof_indices);
                parabolic_system.get_vector("ML").add_vector(M_lumped, parabolic_dof_indices);
                parabolic_system.get_matrix("Ki").add_matrix(Ki, parabolic_dof_indices, parabolic_dof_indices);
                parabolic_system.get_matrix("Ke").add_matrix(Ke, parabolic_dof_indices, parabolic_dof_indices);
                parabolic_system.get_matrix("Kg").add_matrix(Kg, parabolic_dof_indices, parabolic_dof_indices);
                elliptic_system.get_vector("I_extra_stim").add_vector(I_extra_stim, elliptic_dof_indices);
            }
            std::cout << "closing" << std::endl;
            elliptic_system.matrix->close();
            elliptic_system.get_vector("I_extra_stim").close();
            parabolic_system.matrix->close();
            parabolic_system.get_matrix("M").close();
            parabolic_system.get_vector("ML").close();
            parabolic_system.get_matrix("Ki").close();
            parabolic_system.get_matrix("Ke").close();
            parabolic_system.get_matrix("Kg").close();
            std::cout << "done" << std::endl;
            break;
        }
        case TimeIntegrator::SBDF1:
            // Cm * M * ( V^n+1 - Vn ) / dt + Ki * V^n+1 + Ki * Ve^n+1 = -I^n
            // Ki * V^n+1 + Kie * Ve^n+1 = 0
            //         -         -
            //        | Cm /dt * M + Ki,   Ki    |
            //   K  = | Ki,                Ki+Ke |;
            //         -         -
        case TimeIntegrator::SBDF2:
            // Cm * M * ( 3 * V^n+1 - 4 * Vn + V^n-1 ) / (2*dt) + Ki * V^n+1 + Ki * Ve^n+1 = -2*I^n + I^n-1
            // Ki * V^n+1 + Kie * Ve^n+1 = 0
            //         -         -
            //        | 3/2 * Cm /dt * M + Ki,   Ki    |
            //   K  = | Ki,                      Ki+Ke |;
            //         -         -
        case TimeIntegrator::SBDF3:
            // Cm * M * ( 11/6 * V^n+1 - 3 * Vn + 3/2 V^n-1 -1/3 V^n-2) / (dt) + Ki * V^n+1 + Ki * Ve^n+1 = -3*I^n + 3*I^n-1 - I^n-2
            // Ki * V^n+1 + Kie * Ve^n+1 = 0
            //         -         -
            //        | 11/6 * Cm /dt * M + Ki,   Ki    |
            //   K  = | Ki,                      Ki+Ke |;
            //         -         -
        default:
        {
            std::cout << "Assembling matrices for SBDF" << static_cast<int>(time_integrator) << std::endl;

            TransientLinearImplicitSystem &system = es.get_system < TransientLinearImplicitSystem > ("bidomain");
            system.zero_out_matrix_and_rhs = false;
            system.assemble_before_solve = false;
            system.matrix->zero();
            system.get_matrix("M").zero();
            system.get_vector("ML").zero();
            const DofMap &dof_map = system.get_dof_map();
            int V_var_number = system.variable_number("V");
            int Ve_var_number = system.variable_number("Ve");

            DenseSubMatrix<Number> KVV(K), KVVe(K), KVeV(K), KVeVe(K);
            DenseSubMatrix < Number > MVV(M);
            DenseSubVector < Number > ML(M_lumped);

            double coefficient = Cm / timedata.dt;
            if (time_integrator == TimeIntegrator::SBDF2)
                coefficient *= 1.5;
            else if (time_integrator == TimeIntegrator::SBDF3)
                coefficient *= 11 / 6.0;

            for (auto elem : mesh.element_ptr_range())
            {
                dof_map.dof_indices(elem, dof_indices);
                dof_map.dof_indices(elem, parabolic_dof_indices, V_var_number);
                dof_map.dof_indices(elem, elliptic_dof_indices, Ve_var_number);

                fe->reinit(elem);
                int ndofs = dof_indices.size();
                int elliptic_ndofs = elliptic_dof_indices.size();
                int parabolic_ndofs = parabolic_dof_indices.size();

                // resize local elemental matrices
                K.resize(ndofs, ndofs);
                M.resize(ndofs, ndofs);
                M_lumped.resize(ndofs);
                I_extra_stim.resize (ndofs);

                // reposition submatrices
                // Reposition the submatrices...  The idea is this:
                //
                //         -         -
                //        | KVV  KVVe  |
                //   K  = | KVeV KVeVe |;
                //         -         -

                // if we are in the tissue we are using the submatrices
                KVV.reposition(V_var_number * parabolic_ndofs, V_var_number * parabolic_ndofs, parabolic_ndofs, parabolic_ndofs);
                KVVe.reposition(V_var_number * parabolic_ndofs, Ve_var_number * parabolic_ndofs, parabolic_ndofs, elliptic_ndofs);
                KVeV.reposition(Ve_var_number * elliptic_ndofs, V_var_number * elliptic_ndofs, elliptic_ndofs, parabolic_ndofs);
                KVeVe.reposition(Ve_var_number * elliptic_ndofs, Ve_var_number * elliptic_ndofs, elliptic_ndofs, elliptic_ndofs);
                MVV.reposition(V_var_number * parabolic_ndofs, V_var_number * parabolic_ndofs, parabolic_ndofs, parabolic_ndofs);
                ML.reposition(V_var_number * parabolic_ndofs, parabolic_ndofs);

                auto subdomain_id = elem->subdomain_id();
                // if we are in the bath
                if (subdomain_id == static_cast<short unsigned int>(Subdomain::BATH))
                {
                    for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
                    {

                        const Real x = q_point[qp](0);
                        const Real y = q_point[qp](1);
                        const Real z = q_point[qp](2);

                        for (unsigned int i = 0; i != elliptic_ndofs; i++)
                        {
                            for (unsigned int j = 0; j != elliptic_ndofs; j++)
                            {
                                K(i, j) += JxW[qp] * dphi[i][qp] * (sigma_b * dphi[j][qp]);


                                if(zDim == 0){
                                    if( y < stimulus_maxy && y > stimulus_miny && x > stimulus_minx && x < stimulus_maxx ){
                                        //libMesh::out << "top boundary ->    x: " << x << ";      y: " << y << std::endl;
                                        I_extra_stim(i) += JxW[qp]*(     phi[i][qp]*phi[j][qp]*((1./1.)*IstimV)     );
                                    }
                                    else{
                                        I_extra_stim(i) += JxW[qp]*(0.);
                                    }
                                }
                                else{
                                    if( y < stimulus_maxy && y > stimulus_miny && x > stimulus_minx && x < stimulus_maxx && z > stimulus_minz && z < stimulus_maxz ){
                                        //libMesh::out << "top boundary ->    x: " << x << ";      y: " << y << std::endl;
                                        I_extra_stim(i) += JxW[qp]*(     phi[i][qp]*phi[j][qp]*((1./1.)*IstimV)     );
                                    }
                                    else{
                                        I_extra_stim(i) += JxW[qp]*(0.);
                                    }
                                }


                            }
                        }
                    }
                }
                else
                {
                    for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
                    {
                        for (unsigned int i = 0; i != elliptic_ndofs; i++)
                        {
                            for (unsigned int j = 0; j != elliptic_ndofs; j++)
                            {
                                // add mass matrix
                                if (p_order == SECOND) // we cannot do the lumping
                                    KVV(i, j) += JxW[qp] * coefficient * phi[i][qp] * phi[j][qp];
                                else
                                    // we can do the lumping
                                    KVV(i, i) += JxW[qp] * coefficient * phi[i][qp] * phi[j][qp];

                                // stiffness matrix
                                KVV(i, j) += JxW[qp] * dphi[i][qp] * ((sigma_i) * dphi[j][qp]);
                                KVVe(i, j) += JxW[qp] * dphi[i][qp] * ((sigma_i) * dphi[j][qp]);
                                KVeV(i, j) += JxW[qp] * dphi[i][qp] * ((sigma_i) * dphi[j][qp]);
                                KVeVe(i, j) += JxW[qp] * dphi[i][qp] * ((sigma_i + sigma_e) * dphi[j][qp]);

                                // These are useful for the RHS
                                MVV(i, j) += JxW[qp] * phi[i][qp] * phi[j][qp];
                                ML(i) += JxW[qp] * phi[i][qp] * phi[j][qp];
                            }
                        }
                    }
                }
                // boundaries
                if (subdomain_id == static_cast<libMesh::subdomain_id_type>(Subdomain::BATH))
                {
                    for (auto side : elem->side_index_range())
                    {
                        if (elem->neighbor_ptr(side) == nullptr)
                        {
                            auto boundary_id = mesh.get_boundary_info().boundary_id(elem, side);
                            auto it = bc_sidesets.find(static_cast<int>(boundary_id));
                            if (it != bc_sidesets.end())
                            {
                                fe_face->reinit(elem, side);
                                for (unsigned int qp = 0; qp < qface.n_points(); qp++)
                                {
                                    for (unsigned int i = 0; i != elliptic_ndofs; i++)
                                    {
                                        for (unsigned int j = 0; j != elliptic_ndofs; j++)
                                        {
                                            K(i, j) += JxW_face[qp] * penalty * phi_face[j][qp] * phi_face[i][qp];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                system.matrix->add_matrix (K, dof_indices, dof_indices);
                system.get_matrix("M").add_matrix(M, dof_indices, dof_indices);
                system.get_vector("ML").add_vector(M_lumped, dof_indices);
                system.get_vector("I_extra_stim").add_vector(I_extra_stim, dof_indices);
            }
            system.matrix->close();
            system.get_matrix("M").close();
            system.get_vector("ML").close();
            system.get_vector("I_extra_stim").close();
            break;
        }
    }
}

void assemble_forcing_and_eval_error(libMesh::EquationSystems &es, const TimeData &timedata, TimeIntegrator time_integrator, libMesh::Order p_order, const GetPot &data, IonicModel& ionic_model, std::vector<double>& error )
{
    using namespace libMesh;
    // Create vector of BC sidesets ids
    const MeshBase &mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();

    // volume element
    FEType fe_type(p_order);
    std::unique_ptr < FEBase > fe(FEBase::build(dim, fe_type));
    Order qp_order = THIRD;
    if (p_order > 1)
        qp_order = FIFTH;
    libMesh::QGauss qrule(dim, qp_order);
    fe->attach_quadrature_rule(&qrule);

    // quantities for volume integration
    const std::vector<Real> &JxW = fe->get_JxW();
    const std::vector<Point> &q_point = fe->get_xyz();
    const std::vector<std::vector<Real>> &phi = fe->get_phi();

    // define fiber field
    double fx = data("fx", 1.0), fy = data("fy", 0.0), fz = data("fz", 0.0);
    double sx = data("sx", 0.0), sy = data("sy", 1.0), sz = data("sz", 0.0);
    double nx = data("nx", 0.0), ny = data("ny", 0.0), nz = data("nz", 1.0);
    VectorValue<Number>  f0(fx, fy, fz);
    VectorValue<Number>  s0(sx, sy, sz);
    VectorValue<Number>  n0(nx, ny, nz);
    // setup conductivities:
    // Default parameters from
    // Cardiac excitation mechanisms, wavefront dynamics and strength�interval
    // curves predicted by 3D orthotropic bidomain simulations

    // read parameters
    double sigma_f_i = data("sigma_f_i", 2.3172);
    double sigma_s_i = data("sigma_s_i", 0.2435); // sigma_t in the paper
    double sigma_n_i = data("sigma_n_i", 0.0569);
    double sigma_f_e = data("sigma_f_e", 1.5448);
    double sigma_s_e = data("sigma_s_e", 1.0438);  // sigma_t in the paper
    double sigma_n_e = data("sigma_n_e", 0.37222);
    double sigma_b_ie = data("sigma_b", 6.0);
    double chi = data("chi", 1e3);
    double Cm = data("Cm", 1.5);
    double penalty = data("penalty", 1e8);
    double interface_penalty = data("interface_penalty", 1e4);

    // setup tensors parameters
    // f0 \otimes f0
    TensorValue<Number> fof(fx * fx, fx * fy, fx * fz, fy * fx, fy * fy, fy * fz, fz * fx, fz * fy, fz * fz);
    // s0 \otimes s0
    TensorValue<Number> sos(sx * sx, sx * sy, sx * sz, sy * sx, sy * sy, sy * sz, sz * sx, sz * sy, sz * sz);
    // n0 \otimes n0
    TensorValue<Number> non(nx * nx, nx * ny, nx * nz, ny * nx, ny * ny, ny * nz, nz * nx, nz * ny, nz * nz);

    TensorValue<Number> sigma_i = ( sigma_f_i * fof + sigma_s_i * sos + sigma_n_i * non ) /chi;
    TensorValue<Number> sigma_e = ( sigma_f_e * fof + sigma_s_e * sos + sigma_n_e * non) / chi;
    TensorValue<Number> sigma_b(sigma_b_ie/chi, 0.0, 0.0, 0.0, sigma_b_ie/chi, 0.0, 0.0, 0.0, sigma_b_ie/chi);

    DenseVector < Number > Fp;
    DenseVector < Number > Fe;

    double error_vb = 0;
    double error_ve = 0;
    double error_v = 0;

    std::vector < dof_id_type > dof_indices;
    std::vector < dof_id_type > parabolic_dof_indices;
    std::vector < dof_id_type > elliptic_dof_indices;

    double h = 9.536743164062500e-07;
    double dx = 9.536743164062500e-07;

    double t = timedata.time+timedata.dt;
    if( TimeIntegrator::EXPLICIT_INTRACELLULAR == time_integrator || TimeIntegrator::EXPLICIT_INTRACELLULAR == time_integrator)
    {
        t = timedata.time;
    }
    // Assemble matrices differently based on time integrator
    switch (time_integrator)
    {
        case TimeIntegrator::EXPLICIT_EXTRACELLULAR:
        case TimeIntegrator::EXPLICIT_INTRACELLULAR:
        case TimeIntegrator::SEMI_IMPLICIT:
        case TimeIntegrator::SEMI_IMPLICIT_HALF_STEP:
        {
            std::cout << "Assembling matrices for EXPLICIT EXTRACELLULAR: " << static_cast<int>(time_integrator) << std::endl;
            libMesh::LinearImplicitSystem &elliptic_system = es.get_system < libMesh::LinearImplicitSystem > ("elliptic");
            libMesh::TransientLinearImplicitSystem & parabolic_system = es.get_system < libMesh::TransientLinearImplicitSystem > ("parabolic");
            parabolic_system.get_vector("FV").zero();
            elliptic_system.get_vector("FVe").zero();

            const DofMap & elliptic_dof_map = elliptic_system.get_dof_map();
            const DofMap & parabolic_dof_map = parabolic_system.get_dof_map();


            for (auto elem : mesh.element_ptr_range())
            {
                parabolic_dof_map.dof_indices(elem, parabolic_dof_indices);
                elliptic_dof_map.dof_indices(elem, elliptic_dof_indices);
                fe->reinit(elem);
                int elliptic_ndofs = elliptic_dof_indices.size();
                int parabolic_ndofs = parabolic_dof_indices.size();

                // resize local elemental matrices
                Fp.resize(parabolic_ndofs);
                Fe.resize(elliptic_ndofs);

                auto subdomain_id = elem->subdomain_id();

                //std::cout << "Loop over volume: " << std::flush;

                // if we are in the bath
                if (subdomain_id == static_cast<short unsigned int>(Subdomain::BATH))
                {
                    for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
                    {
                        // First Evaluate Error
                        double Vb_h = elliptic_system.point_value(0, q_point[qp], elem);
                        double Vb = exact_solution_Ve_Vb(q_point[qp], timedata.time);
                        error_vb += JxW[qp] * (Vb_h - Vb);
                        // Then evaluate forcing term
                        // evaluate F = - div sigma_b grad Vb
                        // West
                        Vb = exact_solution_Ve_Vb(q_point[qp], t);
                        double VbW = exact_solution_Ve_Vb(q_point[qp]+libMesh::Point(dx, 0, 0), t);
                        // East
                        double VbE = exact_solution_Ve_Vb(q_point[qp]+libMesh::Point(-dx, 0, 0), t);
                        // North
                        double VbN = exact_solution_Ve_Vb(q_point[qp]+libMesh::Point(0, dx, 0),  t);
                        // South
                        double VbS = exact_solution_Ve_Vb(q_point[qp]+libMesh::Point(0, -dx, 0), t);

                        double Vb_xx = ( VbE - 2 * Vb + VbW ) / (dx * dx);
                        double Vb_yy = ( VbN - 2 * Vb + VbS ) / (dx * dx);

                        double F_bath = - sigma_b(0,0) * Vb_xx - sigma_b(1, 1) * Vb_yy;

                        for (unsigned int i = 0; i != elliptic_ndofs; i++)
                        {
                            Fe(i) += JxW[qp] * F_bath * phi[i][qp];
                        }
                    }
                }
                else
                {
                    for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
                    {
                        // First Evaluate Error
                        double V_h = parabolic_system.point_value(0, q_point[qp], elem);
                        double V = exact_solution_V(q_point[qp], timedata.time);
                        error_v += JxW[qp] * (V_h - V);
                        double Ve_h = elliptic_system.point_value(0, q_point[qp], elem);
                        double Ve = exact_solution_Ve(q_point[qp], timedata.time);
                        error_ve += JxW[qp] * (Ve_h - Ve);

                        // Then evaluate forcing term
                        // evaluate F = - div sigma_i grad V - div sigma_ie grad Ve
                        // West
                        V = exact_solution_V(q_point[qp], t);
                        Ve = exact_solution_Ve(q_point[qp], t);
                       double VW = exact_solution_V(q_point[qp]+libMesh::Point(dx, 0, 0), t);
                        double VeW = exact_solution_Ve(q_point[qp]+libMesh::Point(dx, 0, 0), t);
                        // East
                        double VE = exact_solution_V(q_point[qp]+libMesh::Point(-dx, 0, 0), t);
                        double VeE = exact_solution_Ve(q_point[qp]+libMesh::Point(-dx, 0, 0), t);

                        double V_xx = ( VE - 2 * V + VW ) / (dx * dx);
                        double Ve_xx = ( VeE - 2 * Ve + VeW ) / (dx * dx);

                        double F_elliptic = - sigma_i(0,0) * V_xx - ( sigma_i(0, 0) + sigma_e(0, 0) ) * Ve_xx;
                        for (unsigned int i = 0; i != elliptic_ndofs; i++)
                        {
                            Fe(i) += JxW[qp] * F_elliptic * phi[i][qp];
                        }

                        double tph = t+h;
                        double tmh = t-h;
                        double dVdt = 0.5 * ( exact_solution_V(q_point[qp], tph) - exact_solution_V(q_point[qp], tmh) ) / h;
                        double I_ion = ionic_model.iion(V);
                        double F_parabolic = Cm * dVdt + I_ion - sigma_i(0,0) * V_xx - sigma_i(0, 0) * Ve_xx;
                        for (unsigned int i = 0; i != parabolic_ndofs; i++)
                        {
                            Fp(i) += JxW[qp] * F_parabolic * phi[i][qp];
                        }

                    }
                }
                parabolic_system.get_vector("FV").add_vector(Fp, parabolic_dof_indices);
                elliptic_system.get_vector("FVe").add_vector(Fe, elliptic_dof_indices);
            }
            parabolic_system.get_vector("FV").close();
            elliptic_system.get_vector("FVe").close();
            break;
        }
        case TimeIntegrator::SBDF1:
        case TimeIntegrator::SBDF2:
        case TimeIntegrator::SBDF3:
        default:
        {
            TransientLinearImplicitSystem &system = es.get_system < TransientLinearImplicitSystem > ("bidomain");
            system.get_vector("F").zero();
            const DofMap &dof_map = system.get_dof_map();
            int V_var_number = system.variable_number("V");
            int Ve_var_number = system.variable_number("Ve");

            double coefficient = Cm / timedata.dt;
            if (time_integrator == TimeIntegrator::SBDF2)
                coefficient *= 1.5;
            else if (time_integrator == TimeIntegrator::SBDF3)
                coefficient *= 11 / 6.0;

            for (auto elem : mesh.element_ptr_range())
            {
                dof_map.dof_indices(elem, dof_indices);
                dof_map.dof_indices(elem, parabolic_dof_indices, V_var_number);
                dof_map.dof_indices(elem, elliptic_dof_indices, Ve_var_number);

                fe->reinit(elem);
                int ndofs = dof_indices.size();
                int elliptic_ndofs = elliptic_dof_indices.size();
                int parabolic_ndofs = parabolic_dof_indices.size();

                // resize local elemental matrices
                Fp.resize(parabolic_ndofs);
                Fe.resize(elliptic_ndofs);

                auto subdomain_id = elem->subdomain_id();
                // if we are in the bath
                if (subdomain_id == static_cast<short unsigned int>(Subdomain::BATH))
                {
                    for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
                    {
                        // First Evaluate Error
                        double Vb_h = system.point_value(0, q_point[qp], elem);
                        double Vb = exact_solution_Ve_Vb(q_point[qp], timedata.time);
                        error_vb += JxW[qp] * (Vb_h - Vb);
                        // Then evaluate forcing term
                        // evaluate F = - div sigma_b grad Vb
                        Vb = exact_solution_Ve_Vb(q_point[qp], t);
                        // West
                        double VbW = exact_solution_Ve_Vb(q_point[qp]+libMesh::Point(dx, 0, 0), t);
                        // East
                        double VbE = exact_solution_Ve_Vb(q_point[qp]+libMesh::Point(-dx, 0, 0), t);
                        // North
                        double VbN = exact_solution_Ve_Vb(q_point[qp]+libMesh::Point(0, dx, 0), t);
                        // South
                        double VbS = exact_solution_Ve_Vb(q_point[qp]+libMesh::Point(0, -dx, 0), t);

                        double Vb_xx = ( VbE - 2 * Vb + VbW ) / (dx * dx);
                        double Vb_yy = ( VbN - 2 * Vb + VbS ) / (dx * dx);

                        double F_bath = - sigma_b(0,0) * Vb_xx - sigma_b(1, 1) * Vb_yy;

                        for (unsigned int i = 0; i != elliptic_ndofs; i++)
                        {
                            Fe(i) += JxW[qp] * F_bath * phi[i][qp];
                        }
                    }
                }
                else
                {
                    for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
                    {
                        // First Evaluate Error
                        double V_h = system.point_value(V_var_number, q_point[qp], elem);
                        double V = exact_solution_V(q_point[qp], timedata.time);
                        error_v += JxW[qp] * (V_h - V);
                        double Ve_h = system.point_value(Ve_var_number, q_point[qp], elem);
                        double Ve = exact_solution_Ve(q_point[qp], timedata.time);
                        error_ve += JxW[qp] * (Ve_h - Ve);

                        // Then evaluate forcing term
                        // evaluate F = - div sigma_i grad V - div sigma_ie grad Ve
                        V = exact_solution_V(q_point[qp], t);
                        Ve = exact_solution_Ve(q_point[qp], t);
                        // West
                        double VW = exact_solution_V(q_point[qp]+libMesh::Point(dx, 0, 0), t);
                        double VeW = exact_solution_Ve(q_point[qp]+libMesh::Point(dx, 0, 0), t);
                        // East
                        double VE = exact_solution_V(q_point[qp]+libMesh::Point(-dx, 0, 0), t);
                        double VeE = exact_solution_Ve(q_point[qp]+libMesh::Point(-dx, 0, 0), t);

                        double V_xx = ( VE - 2 * V + VW ) / (dx * dx);
                        double Ve_xx = ( VeE - 2 * Ve + VeW ) / (dx * dx);

                        double F_elliptic = - sigma_i(0,0) * V_xx - ( sigma_i(0, 0) + sigma_e(0, 0) ) * Ve_xx;
                        for (unsigned int i = 0; i != elliptic_ndofs; i++)
                        {
                            Fe(i) += JxW[qp] * F_elliptic * phi[i][qp];
                        }

                        double tph = t+h;
                        double tmh = t-h;
                        double dVdt = 0.5 * ( exact_solution_V(q_point[qp], tph) - exact_solution_V(q_point[qp], tmh) ) / h;
                        double I_ion = ionic_model.iion(V);
                        double F_parabolic = Cm * dVdt + I_ion - sigma_i(0,0) * V_xx - sigma_i(0, 0) * Ve_xx;
                        for (unsigned int i = 0; i != parabolic_ndofs; i++)
                        {
                            Fp(i) += JxW[qp] * F_parabolic * phi[i][qp];
                        }
                    }
                }
                system.get_vector("F").add_vector(Fe, elliptic_dof_indices);
                system.get_vector("F").add_vector(Fp, parabolic_dof_indices);
            }
            system.get_vector("F").close();
            break;
        }
    }
    error[0] = std::max(std::sqrt(error_v), error[0]);
    error[1] = std::max(std::sqrt(error_ve), error[1]);
    error[2] = std::max(std::sqrt(error_vb), error[2]);
}


void assemble_rhs(  libMesh::EquationSystems &es,
                    const TimeData &timedata,
                    TimeIntegrator time_integrator,
                    libMesh::Order p_order,
                    const GetPot &data,
                    EquationType type)
{
    using namespace libMesh;
    double Cm = data("Cm", 1.0);
    double chi = data("chi", 1000.0);
    double sigma_s_i = data("sigma_s_i", 0.2435); // sigma_t in the paper
    double sigma_s_e = data("sigma_s_e", 1.0438);  // sigma_t in the paper
    double sigma_b_ie = data("sigma_b", 6.0);
    bool convergence_test = data("convergence_test", false);
    double xDim = data("maxx", 1.) - data("minx", -1.);
    double v0 = data("v0", 0.);
    double v1 = data("v1", 0.05);
    double v2 = data("v2", 1.);
    double kcubic = data("k", 8.);
    //std::cout << " assemble_rhs " << std::endl;
    switch (time_integrator)
    {
        case TimeIntegrator::EXPLICIT_EXTRACELLULAR:
        case TimeIntegrator::EXPLICIT_INTRACELLULAR:
        case TimeIntegrator::SEMI_IMPLICIT:
        case TimeIntegrator::SEMI_IMPLICIT_HALF_STEP:
         {
            const MeshBase& mesh = es.get_mesh();
            libMesh::LinearImplicitSystem &elliptic_system = es.get_system < libMesh::LinearImplicitSystem > ("elliptic");
            libMesh::TransientLinearImplicitSystem & parabolic_system = es.get_system < libMesh::TransientLinearImplicitSystem > ("parabolic");
            const DofMap & elliptic_dof_map = elliptic_system.get_dof_map();
            const DofMap & parabolic_dof_map = parabolic_system.get_dof_map();
            std::vector < dof_id_type > parabolic_dof_indices;
            std::vector < dof_id_type > elliptic_dof_indices;

            if(type == EquationType::PARABOLIC)
            {
                parabolic_system.get_vector("aux1").zero();
                parabolic_system.rhs->zero();

                // Transfer Ve from elliptic to parabolic
                for (auto node : mesh.node_ptr_range())
                {
                    parabolic_dof_map.dof_indices(node, parabolic_dof_indices);
                    elliptic_dof_map.dof_indices(node, elliptic_dof_indices);
                    int elliptic_ndofs = elliptic_dof_indices.size();
                    int parabolic_ndofs = parabolic_dof_indices.size();

                    if (elliptic_ndofs == parabolic_ndofs)
                    {
                        double ven = (*elliptic_system.solution)(elliptic_dof_indices[0]);
                        parabolic_system.get_vector("aux1").set(parabolic_dof_indices[0], ven);
                    }
                }
                parabolic_system.get_vector("aux1").close();

                if(time_integrator == TimeIntegrator::EXPLICIT_EXTRACELLULAR)
                {
                    // Assemble RHS:
                    parabolic_system.rhs->add_vector(parabolic_system.get_vector("aux1"), parabolic_system.get_matrix("Ke"));
                    parabolic_system.rhs->add_vector(parabolic_system.get_vector("aux1"), parabolic_system.get_matrix("Kg"));
                    parabolic_system.rhs->add_vector(*parabolic_system.solution, parabolic_system.get_matrix("Kg"));
                }
                else if(time_integrator == TimeIntegrator::EXPLICIT_INTRACELLULAR)
                {
                    // Assemble RHS:
                    parabolic_system.rhs->add_vector(parabolic_system.get_vector("aux1"), parabolic_system.get_matrix("Ki"));
                    parabolic_system.rhs->add_vector(*parabolic_system.solution, parabolic_system.get_matrix("Ki"));
                    parabolic_system.rhs->scale(-1.0);
                }
                else // SEMI IMPLICIT
                {
                    // Assemble RHS:
                    parabolic_system.rhs->add_vector(parabolic_system.get_vector("aux1"), parabolic_system.get_matrix("Ki"));
                    parabolic_system.rhs->scale(-1.0);
                }
                parabolic_system.get_vector("aux1").zero();
                parabolic_system.get_vector("aux1").add(1.0, parabolic_system.get_vector("In"));
                // add  M * In
                parabolic_system.rhs->add_vector(parabolic_system.get_vector("aux1"), parabolic_system.get_matrix("M"));
                parabolic_system.rhs->add(parabolic_system.get_vector("FV"));

                if(time_integrator == TimeIntegrator::SEMI_IMPLICIT ||
                   time_integrator == TimeIntegrator::SEMI_IMPLICIT_HALF_STEP)
                {
                    parabolic_system.get_vector("aux1").zero();
                    parabolic_system.get_vector("aux1").pointwise_mult(*parabolic_system.solution, parabolic_system.get_vector("ML"));
                    *parabolic_system.rhs += parabolic_system.get_vector("aux1");
                }
                else
                {
                    (*parabolic_system.rhs) /= parabolic_system.get_vector("ML");
                }
                // add forcing

            }
            else
            {
                parabolic_system.get_vector("aux1").zero();
                elliptic_system.rhs->zero();

                parabolic_system.get_vector("aux1").add_vector(*parabolic_system.solution, parabolic_system.get_matrix("Ki"));
                // Transfer KiV from parabolic to elliptic
                for (auto node : mesh.node_ptr_range())
                {
                    parabolic_dof_map.dof_indices(node, parabolic_dof_indices);
                    elliptic_dof_map.dof_indices(node, elliptic_dof_indices);
                    int elliptic_ndofs = elliptic_dof_indices.size();
                    int parabolic_ndofs = parabolic_dof_indices.size();

                    if (elliptic_ndofs == parabolic_ndofs)
                    {
                        double KiV = parabolic_system.get_vector("aux1")(parabolic_dof_indices[0]);
                        elliptic_system.rhs->set(elliptic_dof_indices[0], -KiV);
                    }
                }
                elliptic_system.rhs->add(elliptic_system.get_vector("FVe"));

            }
            break;
        }
        case TimeIntegrator::SBDF3:
            // Cm * M * ( 11/6 * V^n+1 - 3 * Vn + 3/2 V^n-1 -1/3 V^n-2) / (dt) + Ki * V^n+1 + Ki * Ve^n+1 = -3*I^n + 3*I^n-1 - I^n-2
            // Ki * V^n+1 + Kie * Ve^n+1 = 0            //
            //   RHS = 3 Cm /dt * M * Vn - 3/2 Cm /dt * M * Vnm1 +Cm/dt/3 M Vnm2 - 3 * M In + 3 Inm1 - Inm2

        {
            TransientLinearImplicitSystem &system = es.get_system < TransientLinearImplicitSystem > ("bidomain");
            //system.get_vector("In").print();
            system.rhs->zero();
            system.get_vector("aux1").zero();
            // eval: 3 Cm /dt * M * Vn
            system.get_vector("aux1").add(3.0*Cm/timedata.dt, *system.old_local_solution);
            // add: -1.5 Cm /dt * M * Vnm1
            system.get_vector("aux1").add(-1.5*Cm/timedata.dt, *system.older_local_solution);
            // add: +1/3 Cm /dt * M * Vnm2
            system.get_vector("aux1").add(Cm/(3*timedata.dt), system.get_vector("Vnm2"));
            //system.get_vector("aux1").print();

            if(p_order == SECOND)
            {
                system.rhs->add_vector(system.get_vector("aux1"), system.get_matrix("M"));
            }
            else
            {
                system.rhs->pointwise_mult(system.get_vector("aux1"), system.get_vector("ML"));
            }
            // add  M * (2*In-Inm1)
            system.get_vector("aux1").zero();
            system.get_vector("aux1").add(3.0, system.get_vector("In"));
            system.get_vector("aux1").add(-3.0, system.get_vector("Inm1"));
            system.get_vector("aux1").add(1.0, system.get_vector("Inm2"));
            //system.get_vector("aux1").print();

            // add  M * In
            system.rhs->add_vector(system.get_vector("aux1"), system.get_matrix("M"));
            // add forcing
            if(convergence_test){
                system.get_vector("ForcingConv").zero();
                ForcingTermConvergence ( es, timedata.dt, chi, Cm, sigma_s_i, sigma_s_e, sigma_b_ie, timedata.time, xDim, v0, v1, v2, kcubic);
                system.rhs->add(system.get_vector("ForcingConv"));
            }
            //system.rhs->add(system.get_vector("F"));


            break;
        }

        case TimeIntegrator::SBDF2:
            // Cm * M * ( 3 * V^n+1 - 4 * Vn + V^n-1 ) / (2*dt) + Ki * V^n+1 + Ki * Ve^n+1 = -2*I^n + I^n-1
            // Ki * V^n+1 + Kie * Ve^n+1 = 0
            //
            //   RHS = 2 Cm /dt * M * Vn - 1/2 Cm /dt * M * Vnm1 + 2 * M In - Inm1
            // Cm /dt M ( 2 * Vn - 1/2  Vnm1)
        {
            TransientLinearImplicitSystem &system = es.get_system < TransientLinearImplicitSystem > ("bidomain");
            system.rhs->zero();
            system.get_vector("aux1").zero();
            //system.get_vector("In").print();
            // eval: 2 Cm /dt * M * Vn
            system.get_vector("aux1").add(2*Cm/timedata.dt, *system.old_local_solution);
            // add: -0.5 Cm /dt * M * Vnm1
            system.get_vector("aux1").add(-.5*Cm/timedata.dt, *system.older_local_solution);
            //system.get_vector("aux1").print();
            if(p_order == SECOND)
            {
                system.rhs->add_vector(system.get_vector("aux1"), system.get_matrix("M"));
            }
            else
            {
                system.rhs->pointwise_mult(system.get_vector("aux1"), system.get_vector("ML"));
            }
            // add  M * (2*In-Inm1)
            system.get_vector("aux1").zero();
            system.get_vector("aux1").add(2.0, system.get_vector("In"));
            system.get_vector("aux1").add(-1.0, system.get_vector("Inm1"));
            //system.get_vector("aux1").print();
            // add  M * In
            system.rhs->add_vector(system.get_vector("aux1"), system.get_matrix("M"));
            //system.get_vector("aux1").pointwise_mult(system.get_vector("aux1"), system.get_vector("ML"));
            //*system.rhs += system.get_vector("aux1");
            // add forcing
            if(convergence_test){
                system.get_vector("ForcingConv").zero();
                ForcingTermConvergence ( es, timedata.dt, chi, Cm, sigma_s_i, sigma_s_e, sigma_b_ie, timedata.time, xDim, v0, v1, v2, kcubic);
                system.rhs->add(system.get_vector("ForcingConv"));
            }
            //system.rhs->add(system.get_vector("F"));

            break;
        }
        case TimeIntegrator::SBDF1:
            // Cm * M * ( V^n+1 - Vn ) / dt + Ki * V^n+1 + Ki * Ve^n+1 = -I^n
            // Ki * V^n+1 + Kie * Ve^n+1 = 0
            //
            //   RHS = Cm /dt * M * Vn + M In
        default:
        {

            TransientLinearImplicitSystem &system = es.get_system < TransientLinearImplicitSystem > ("bidomain");
            system.rhs->zero();
            system.get_vector("aux1").zero();

            // eval: Cm /dt * M * Vn
            system.get_vector("aux1").add(Cm/timedata.dt, *system.old_local_solution);
            if(p_order == SECOND)
            {
                system.rhs->add_vector(system.get_vector("aux1"), system.get_matrix("M"));
            }
            else
            {
                system.rhs->pointwise_mult(system.get_vector("aux1"), system.get_vector("ML"));
            }
            // add  M * In
            system.get_vector("aux1").zero();
            system.get_vector("aux1").add(1.0, system.get_vector("In"));
            // add  M * In
            system.rhs->add_vector(system.get_vector("aux1"), system.get_matrix("M"));
            // add forcing
            if(convergence_test){
                system.get_vector("ForcingConv").zero();
                ForcingTermConvergence ( es, timedata.dt, chi, Cm, sigma_s_i, sigma_s_e, sigma_b_ie, timedata.time, xDim, v0, v1, v2, kcubic);
                system.rhs->add(system.get_vector("ForcingConv"));
            }
            //system.rhs->add(system.get_vector("F"));


            break;
        }
    }
}




// TODO: update for SBDF2 and SBDF3
void solve_ionic_model_evaluate_ionic_currents(libMesh::EquationSystems &es,
                                               IonicModel& ionic_model,
                                               Pacing& pacing,
                                               TimeData& datatime,
                                               TimeIntegrator &time_integrator)
{
    using namespace libMesh;
    const MeshBase& mesh = es.get_mesh();
    if( time_integrator == TimeIntegrator::EXPLICIT_EXTRACELLULAR ||
        time_integrator == TimeIntegrator::EXPLICIT_INTRACELLULAR ||
        time_integrator == TimeIntegrator::SEMI_IMPLICIT ||
        time_integrator == TimeIntegrator::SEMI_IMPLICIT_HALF_STEP )
    {
        libMesh::LinearImplicitSystem &elliptic_system = es.get_system < libMesh::LinearImplicitSystem > ("elliptic");
        libMesh::TransientLinearImplicitSystem & parabolic_system = es.get_system < libMesh::TransientLinearImplicitSystem > ("parabolic");
        const DofMap & elliptic_dof_map = elliptic_system.get_dof_map();
        const DofMap & parabolic_dof_map = parabolic_system.get_dof_map();
        std::vector < dof_id_type > parabolic_dof_indices;
        std::vector < dof_id_type > elliptic_dof_indices;

        for (auto node : mesh.node_ptr_range())
        {
            parabolic_dof_map.dof_indices(node, parabolic_dof_indices);
            elliptic_dof_map.dof_indices(node, elliptic_dof_indices);
            int elliptic_ndofs = elliptic_dof_indices.size();
            int parabolic_ndofs = parabolic_dof_indices.size();

            if (elliptic_ndofs == parabolic_ndofs)
            {
                double stimulus = pacing.istim(*node, datatime.time);
                double vn = (*parabolic_system.solution)(parabolic_dof_indices[0]);
                double i_ion = ionic_model.iion(vn);
                double i_tot = i_ion + stimulus;
                parabolic_system.get_vector("In").set(parabolic_dof_indices[0], i_tot);
            }
        }
        parabolic_system.get_vector("In").close();
    }
    else
    {
        TransientLinearImplicitSystem &system = es.get_system < TransientLinearImplicitSystem > ("bidomain");
        const DofMap &dof_map = system.get_dof_map();
        int V_var_number = system.variable_number("V");
        int Ve_var_number = system.variable_number("Ve");
        std::vector < dof_id_type > dof_indices;
        std::vector < dof_id_type > parabolic_dof_indices;
        std::vector < dof_id_type > elliptic_dof_indices;

        for (auto node : mesh.node_ptr_range())
        {
            dof_map.dof_indices(node, parabolic_dof_indices, V_var_number);
            dof_map.dof_indices(node, elliptic_dof_indices, Ve_var_number);
            int elliptic_ndofs = elliptic_dof_indices.size();
            int parabolic_ndofs = parabolic_dof_indices.size();

            if (elliptic_ndofs == parabolic_ndofs)
            {
                double stimulus = pacing.istim(*node, datatime.time);
                double vn = (*system.solution)(parabolic_dof_indices[0]);
                double i_ion = ionic_model.iion(vn);
                double i_tot = i_ion + stimulus;
                system.get_vector("In").set(parabolic_dof_indices[0], i_tot);
            }
        }
        system.get_vector("In").close();
    }
}


// read BC sidesets from string: e.g. bc = "1 2 3 5", or bc = "1, 55, 2, 33"
void read_bc_list(std::string &bc, std::set<int> &bc_sidesets)
{
    std::string number;
    for (auto it = bc.begin(); it != bc.end(); it++)
    {
        auto character = *it;
        if (std::isdigit(character))
            number += character;
        else
        {
            if (number.size() > 0)
            {
                bc_sidesets.insert(std::stoi(number));
                number.clear();
            }
        }
        if (it + 1 == bc.end())
        {
            bc_sidesets.insert(std::stoi(number));
        }
    }
    std::cout << "BC sideset list: " << std::flush;
    for (auto &&sideset : bc_sidesets)
        std::cout << sideset << ", " << std::flush;
    std::cout << std::endl;
}


double initial_condition_V(const libMesh::Point& p, const double time)
{
    return 0;
}
double initial_condition_Ve(const libMesh::Point& p, const double time)
{
    return 0;
}
double exact_solution_V(const libMesh::Point& p, const double time)
{
    double x = p(0), y = p(1), z = p(2);
    double x0 = -.5;
    double c = .125;
    double sigma = .125;
    double delta = .1;
    double alpha = 50.;
    double dummy = x0 - x + c*time;
    double V = tanh(alpha*(dummy))*.5 + .5;
    return V;
}

double exact_solution_Ve(const libMesh::Point& p, const double time)
{
    double x = p(0), y = p(1), z = p(2);
    double L = 1.5;
    double x0 = -.5;
    double c = .125;
    double sigma = .125;
    double delta = .1;
    double alpha = 50.;
    double dummy = x0 - x + c*time;

    double sigma6 = sigma * sigma * sigma * sigma * sigma * sigma;
    double innerPow1 = delta - dummy;
    double innerPow2 = delta + dummy;

    double exp1 = exp(-pow((innerPow1),6)/sigma6);
    double exp2 = exp(-pow((innerPow2),6)/sigma6);
    double Ve = exp1 - exp2;
    return Ve;
}
double exact_solution_Ve_Vb(const libMesh::Point& p, const double time)
{
    double Ve = exact_solution_Ve(p, time);
    double L = 1.5;
    double x = p(0), y = p(1), z = p(2);
    double a1 = 8.0/(L*L);
    double b1 = 16.0/L;
    double c1 = 8.0;
    double a2 = -8.0/(L*L);
    double b2 = -8.0/(L);
    double c2 = -1.0;

    if( y > 0 )
    {
      b1 = -b1;
      b2 = -b2;
    }

    double g1 = a1*y*y + b1*y + c1;
    double g2 = a2*y*y + b2*y + c2;
    double g = 1.0;
    if(y <= -0.75*L)
    {
          g = g1;
    }
    else if(y > -0.75 * L && y < -0.5 * L)
    {
          g = g2;
    }
    else if(y >= 0.75 * L)
    {
          g = g1;
    }
    else if(y < 0.75 * L && y > 0.5 * L)
    {
          g = g2;
    }
    double Vb = Ve * g;
    return Vb;
}


void exact_solution_monolithic(libMesh::DenseVector<libMesh::Number>& output, const libMesh::Point& p, const double time)
{
    output(0) = exact_solution_V(p, time);
    output(1) = exact_solution_Ve_Vb(p, time);
}






void SolverRecovery (EquationSystems & es, const GetPot &data, TimeData& datatime)
{
  
  auto time = datatime.time;
  const Real dt = datatime.dt;
  std::string integrator = data("integrator", "SBDF1");
  double zDim = data("maxz", 0.) - data("minz", 0.);
  double xDim = data("maxx", 0.) - data("minx", 0.);
  int nelx = data("nelx", 40);
  double IstimD = data("stimulus_duration", 2.);
  double IstimV = data("stimulus_amplitude", -1.);
  double tissue_maxx = data("tissue_maxx", .5);
  double tissue_minx = data("tissue_minx", .5);
  double tissue_maxy = data("tissue_maxy", .5);
  double tissue_miny = data("tissue_miny", .5);
  double tissue_maxz = data("tissue_maxz", .5);
  double tissue_minz = data("tissue_minz", .5);
  double stimulus_maxx = data("stimulus_maxx", .5);
  double stimulus_maxy = data("stimulus_maxy", .5);
  double stimulus_minx = data("stimulus_minx", -.5);
  double stimulus_miny = data("stimulus_miny", .85);
  double stimulus_maxz = data("stimulus_maxz", 0.0);
  double stimulus_minz = data("stimulus_minz", 0.0);
  double stimulus_start_time = data("stimulus_start_time", 0.);
  bool convergence_test = data("convergence_test", false);
  double u0 = data("v0", 0.);//-85;
  double u1 = data("v1",.05);//-57.6;
  double u2 = data("v2",1.0);//30;
  double kcubic = data("k", 8.);
  std::string StimPlace = data("stimulus_type", "Transmembrane");
  int SpiralBool = data("SpiralBool", 0);
  
  TransientExplicitSystem & Rsystem = es.get_system<TransientExplicitSystem> ("Recovery");
  Rsystem.solution -> close();
  Rsystem.update();
  const unsigned int v_var = Rsystem.variable_number ("v");
  const unsigned int w_var = Rsystem.variable_number ("w");
  const unsigned int s_var = Rsystem.variable_number ("s");
  const DofMap & dof_map2 = Rsystem.get_dof_map();

  MeshBase & mesh = es.get_mesh();

  const unsigned int dim = mesh.mesh_dimension();

  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices2;

  double v_new, w_new, s_new, u_old, v_old, w_old, s_old, u_old_old, v_old_old, w_old_old, s_old_old, u_old_old_old, v_old_old_old, w_old_old_old, s_old_old_old;

  double r_new;

  //Bsystem.get_vector("I_ion").zero();
  //Rsystem.solution->zero();

  //Bsystem.get_vector("I_ion").close();

  //libMesh::out << "I AM HERE too INSIDE SOLVER RECOVERY" << std::endl;

  double tau_v_plus = 3.33;
  double tau_v1_minus = 19.2;
  double tau_v2_minus = 10.0;
  double tau_w_plus = 160.0;
  double tau_w1_minus = 75.0;
  double tau_w2_minus = 75.0;
  double tau_d = .065;
  double tau_si = 31.8364;
  double tau_o = 39.0;
  double tau_a = .009;
  double u_c = .23;
  double u_v = .055;
  double u_w = .146;
  double u_o = 0.0;
  double u_m = 1.0;
  double u_csi = .8;
  double u_so = .3;
  double r_s_plus = .02;
  double r_s_minus = 1.2;
  double k = 3.0;
  double a_so = .115;
  double b_so = .84;
  double c_so = .02;


  double phi [1000] = {};
  double x1 [1000] = {};
  double y1 [1000] = {};
  double threshSpiral = (xDim*2.5/ (double)nelx);

  if(SpiralBool == 1 || SpiralBool == 2){
      //spiral wave change for variables: parameter set 1 Fenton et al 2002
      tau_v_plus = 3.33;
      tau_v1_minus = 19.6;
      tau_v2_minus = 1000.0;
      tau_w_plus = 667.0;
      tau_w1_minus = 11.0;
      tau_w2_minus = 11.0;
      tau_d = .25;
      tau_si = 45.;
      tau_o = 8.3;
      tau_a = .02; //tau_a = 1/tau_r
      k = 10.;
      u_csi = .85;
      u_c = .13;
      u_v = .055;

    }


  
  if(integrator.compare(0,4,"SBDF") == 0){

    TransientLinearImplicitSystem &Bsystem = es.get_system <TransientLinearImplicitSystem> ("bidomain");
    Bsystem.update();
    const unsigned int u_var = Bsystem.variable_number ("V");
    const DofMap & dof_map = Bsystem.get_dof_map();


    for(const auto & node : mesh.local_node_ptr_range()){

        dof_map.dof_indices (node, dof_indices);
        dof_map2.dof_indices (node, dof_indices2);

        const Real x = (*node)(0);
        const Real y = (*node)(1);
        const Real z = (*node)(2);

        if(dof_indices2.size() > 0){


            if(time == dt){
                v_old = 1.;
                w_old = 1.;
                s_old = 0.;
                u_old = (*Bsystem.current_local_solution)  (dof_indices[u_var]);

                if(SpiralBool == 1){
                    if( y < 0){
                        v_old = 0.;
                        w_old = 0.;
                        s_old = 1.;
                        u_old = 1.;
                    }
                }

                else if(SpiralBool == 2){

                    double a = .5;
                    double k = .09;
                    int thickness = 15;
                    int finalDim = thickness*1000; //15,000
                    double bathRegion = xDim/2.0 - tissue_maxx;
                    double factorAxis = 3.0*(xDim/2.-bathRegion);
                    phi[0] = 0.;
                    x1[0] = -a*exp(k*phi[0])*cos(phi[0]);
                    y1[0] = -a*exp(k*phi[0])*sin(phi[0]);
                    for(int i = 1; i < 1000; i++){
                        phi[i] = phi[i-1] + ((factorAxis*acos(-1) - phi[0])/1000.);
                        x1[i] = -a*exp(k*phi[i])*cos(phi[i]);
                        y1[i] = -a*exp(k*phi[i])*sin(phi[i]);
                    }
                    for(int i = 0; i < 1000; i++){
                        if( std::abs(x - x1[i]) <= threshSpiral && std::abs(y - y1[i]) <= threshSpiral){
                            //conditions in notes seen from paraview
                            /*
                             * initial conditions for multiple spirals to happen:
                                u - spiral of 1. and rest of 0
                                v - inverse of spiral of 1. and rest of 0
                                w - spiral of .8 and rest of 1.
                                s - same spiral of .1 and rest of 0
                            */
                            v_old = 0.;
                            w_old = 0.;
                            s_old = 1.;
                            u_old = 1.;
                            //libMesh::out << "HERE --> x = " << x << ";     y = " << y << ";    with threshold of the spiral: " << threshSpiral << std::endl;
                        }
                    }
                }


            }
            else{
                v_old = (*Rsystem.current_local_solution) (dof_indices2[v_var]);
                w_old = (*Rsystem.current_local_solution) (dof_indices2[w_var]);        
                s_old = (*Rsystem.current_local_solution) (dof_indices2[s_var]);        
                u_old = (*Bsystem.current_local_solution)  (dof_indices[u_var]);
                u_old_old = (Bsystem.get_vector("Vnm1"))  (dof_indices[0]);
                u_old_old_old = (Bsystem.get_vector("Vnm2"))  (dof_indices[0]);
                v_old_old = (Rsystem.get_vector("v_prev")) (dof_indices2[0]);
                v_old_old_old = (Rsystem.get_vector("v_prev_prev")) (dof_indices2[0]);
                w_old_old = (Rsystem.get_vector("w_prev")) (dof_indices2[0]);
                w_old_old_old = (Rsystem.get_vector("w_prev_prev")) (dof_indices2[0]);
                s_old_old = (Rsystem.get_vector("s_prev")) (dof_indices2[0]);
                s_old_old_old = (Rsystem.get_vector("s_prev_prev")) (dof_indices2[0]);
            }


            if(integrator.compare(0,5,"SBDF3") == 0){

                if(time == dt){
                  double RhsV = ( ((1.0 - HofX(u_old,u_c))*(1.0 - v_old))/(tau_v2_minus*HofX(u_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old,u_v)))) - ((HofX(u_old,u_c)*v_old)/(tau_v_plus));
                  double RhsW = ( ((1.0 - HofX(u_old,u_c))*(1.0 - w_old))/(tau_w2_minus*HofX(u_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old,u_w)))) - ((HofX(u_old,u_c)*w_old)/(tau_w_plus));
                  double RhsS = (r_s_plus*(HofX(u_old,u_c)) + r_s_minus*(1.0 - HofX(u_old,u_c))) * (.5*(1.0 + tanh(k*(u_old - u_csi))) - s_old);

                  v_new = v_old + dt*RhsV;
                  Rsystem.solution -> set(dof_indices2[v_var],v_new);
                  w_new = w_old + dt*RhsW;
                  Rsystem.solution -> set(dof_indices2[w_var],w_new);
                  s_new = s_old + dt*RhsS;
                  Rsystem.solution -> set(dof_indices2[s_var],s_new);
                }
                else if(time == 2.*dt){
                  double RhsV = ( ((1.0 - HofX(u_old,u_c))*(1.0 - v_old))/(tau_v2_minus*HofX(u_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old,u_v)))) - ((HofX(u_old,u_c)*v_old)/(tau_v_plus));
                  double RhsW = ( ((1.0 - HofX(u_old,u_c))*(1.0 - w_old))/(tau_w2_minus*HofX(u_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old,u_w)))) - ((HofX(u_old,u_c)*w_old)/(tau_w_plus));
                  double RhsS = (r_s_plus*(HofX(u_old,u_c)) + r_s_minus*(1.0 - HofX(u_old,u_c))) * (.5*(1.0 + tanh(k*(u_old - u_csi))) - s_old);

                  double RhsV_old = ( ((1.0 - HofX(u_old_old,u_c))*(1.0 - v_old_old))/(tau_v2_minus*HofX(u_old_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old_old,u_v)))) - ((HofX(u_old_old,u_c)*v_old_old)/(tau_v_plus));
                  double RhsW_old = ( ((1.0 - HofX(u_old_old,u_c))*(1.0 - w_old_old))/(tau_w2_minus*HofX(u_old_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old_old,u_w)))) - ((HofX(u_old_old,u_c)*w_old_old)/(tau_w_plus));
                  double RhsS_old = (r_s_plus*(HofX(u_old_old,u_c)) + r_s_minus*(1.0 - HofX(u_old_old,u_c))) * (.5*(1.0 + tanh(k*(u_old_old - u_csi))) - s_old_old);


                  v_new = (4.0/3.)*v_old - (1.0/3.)*v_old_old + (4.0/3.)*dt*RhsV -(2.0/3.)*dt*RhsV_old;
                  Rsystem.solution -> set(dof_indices2[v_var],v_new);
                  Rsystem.get_vector("v_prev").set(dof_indices2[0],v_old);
                  w_new = (4.0/3.)*w_old - (1.0/3.)*w_old_old + (4.0/3.)*dt*RhsW -(2.0/3.)*dt*RhsW_old;
                  Rsystem.get_vector("w_prev").set(dof_indices2[0],w_old);
                  Rsystem.solution -> set(dof_indices2[w_var],w_new);
                  s_new = (4.0/3.)*s_old - (1.0/3.)*s_old_old + (4.0/3.)*dt*RhsS -(2.0/3.)*dt*RhsS_old;
                  Rsystem.get_vector("s_prev").set(dof_indices2[0],s_old);
                  Rsystem.solution -> set(dof_indices2[s_var],s_new);
                }
                else{
                  double RhsV = ( ((1.0 - HofX(u_old,u_c))*(1.0 - v_old))/(tau_v2_minus*HofX(u_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old,u_v)))) - ((HofX(u_old,u_c)*v_old)/(tau_v_plus));
                  double RhsW = ( ((1.0 - HofX(u_old,u_c))*(1.0 - w_old))/(tau_w2_minus*HofX(u_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old,u_w)))) - ((HofX(u_old,u_c)*w_old)/(tau_w_plus));
                  double RhsS = (r_s_plus*(HofX(u_old,u_c)) + r_s_minus*(1.0 - HofX(u_old,u_c))) * (.5*(1.0 + tanh(k*(u_old - u_csi))) - s_old);

                  double RhsV_old = ( ((1.0 - HofX(u_old_old,u_c))*(1.0 - v_old_old))/(tau_v2_minus*HofX(u_old_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old_old,u_v)))) - ((HofX(u_old_old,u_c)*v_old_old)/(tau_v_plus));
                  double RhsW_old = ( ((1.0 - HofX(u_old_old,u_c))*(1.0 - w_old_old))/(tau_w2_minus*HofX(u_old_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old_old,u_w)))) - ((HofX(u_old_old,u_c)*w_old_old)/(tau_w_plus));
                  double RhsS_old = (r_s_plus*(HofX(u_old_old,u_c)) + r_s_minus*(1.0 - HofX(u_old_old,u_c))) * (.5*(1.0 + tanh(k*(u_old_old - u_csi))) - s_old_old);

                  double RhsV_old_old = ( ((1.0 - HofX(u_old_old_old,u_c))*(1.0 - v_old_old_old))/(tau_v2_minus*HofX(u_old_old_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old_old_old,u_v)))) - ((HofX(u_old_old_old,u_c)*v_old_old_old)/(tau_v_plus));
                  double RhsW_old_old = ( ((1.0 - HofX(u_old_old_old,u_c))*(1.0 - w_old_old_old))/(tau_w2_minus*HofX(u_old_old_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old_old_old,u_w)))) - ((HofX(u_old_old_old,u_c)*w_old_old_old)/(tau_w_plus));
                  double RhsS_old_old = (r_s_plus*(HofX(u_old_old_old,u_c)) + r_s_minus*(1.0 - HofX(u_old_old_old,u_c))) * (.5*(1.0 + tanh(k*(u_old_old_old - u_csi))) - s_old_old_old);

                  v_new = (18.0/11.)*v_old - (18.0/22.)*v_old_old + (6.0/33.)*v_old_old_old + (18.0/11.)*dt*RhsV - (18.0/11.)*dt*RhsV_old + (6.0/11.)*dt*RhsV_old_old;
                  Rsystem.solution -> set(dof_indices2[v_var],v_new);
                  Rsystem.get_vector("v_prev").set(dof_indices2[0],v_old);
                  Rsystem.get_vector("v_prev_prev").set(dof_indices2[0],v_old_old);
                  w_new = (18.0/11.)*w_old - (18.0/22.)*w_old_old + (6.0/33.)*w_old_old_old + (18.0/11.)*dt*RhsW - (18.0/11.)*dt*RhsW_old + (6.0/11.)*dt*RhsW_old_old;
                  Rsystem.get_vector("w_prev").set(dof_indices2[0],w_old);
                  Rsystem.get_vector("w_prev_prev").set(dof_indices2[0],w_old_old);
                  Rsystem.solution -> set(dof_indices2[w_var],w_new);
                  s_new = (18.0/11.)*s_old - (18.0/22.)*s_old_old + (6.0/33.)*s_old_old_old + (18.0/11.)*dt*RhsS - (18.0/11.)*dt*RhsS_old + (6.0/11.)*dt*RhsS_old_old;
                  Rsystem.get_vector("s_prev").set(dof_indices2[0],s_old);
                  Rsystem.get_vector("s_prev_prev").set(dof_indices2[0],s_old_old);
                  Rsystem.solution -> set(dof_indices2[s_var],s_new);
                }

            }
            else if(integrator.compare(0,5,"SBDF2") == 0){

                if(time == dt){
                  double RhsV = ( ((1.0 - HofX(u_old,u_c))*(1.0 - v_old))/(tau_v2_minus*HofX(u_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old,u_v)))) - ((HofX(u_old,u_c)*v_old)/(tau_v_plus));
                  double RhsW = ( ((1.0 - HofX(u_old,u_c))*(1.0 - w_old))/(tau_w2_minus*HofX(u_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old,u_w)))) - ((HofX(u_old,u_c)*w_old)/(tau_w_plus));
                  double RhsS = (r_s_plus*(HofX(u_old,u_c)) + r_s_minus*(1.0 - HofX(u_old,u_c))) * (.5*(1.0 + tanh(k*(u_old - u_csi))) - s_old);

                  v_new = v_old + dt*RhsV;
                  Rsystem.solution -> set(dof_indices2[v_var],v_new);
                  w_new = w_old + dt*RhsW;
                  Rsystem.solution -> set(dof_indices2[w_var],w_new);
                  s_new = s_old + dt*RhsS;
                  Rsystem.solution -> set(dof_indices2[s_var],s_new);
                }
                else{
                  double RhsV = ( ((1.0 - HofX(u_old,u_c))*(1.0 - v_old))/(tau_v2_minus*HofX(u_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old,u_v)))) - ((HofX(u_old,u_c)*v_old)/(tau_v_plus));
                  double RhsW = ( ((1.0 - HofX(u_old,u_c))*(1.0 - w_old))/(tau_w2_minus*HofX(u_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old,u_w)))) - ((HofX(u_old,u_c)*w_old)/(tau_w_plus));
                  double RhsS = (r_s_plus*(HofX(u_old,u_c)) + r_s_minus*(1.0 - HofX(u_old,u_c))) * (.5*(1.0 + tanh(k*(u_old - u_csi))) - s_old);

                  double RhsV_old = ( ((1.0 - HofX(u_old_old,u_c))*(1.0 - v_old_old))/(tau_v2_minus*HofX(u_old_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old_old,u_v)))) - ((HofX(u_old_old,u_c)*v_old_old)/(tau_v_plus));
                  double RhsW_old = ( ((1.0 - HofX(u_old_old,u_c))*(1.0 - w_old_old))/(tau_w2_minus*HofX(u_old_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old_old,u_w)))) - ((HofX(u_old_old,u_c)*w_old_old)/(tau_w_plus));
                  double RhsS_old = (r_s_plus*(HofX(u_old_old,u_c)) + r_s_minus*(1.0 - HofX(u_old_old,u_c))) * (.5*(1.0 + tanh(k*(u_old_old - u_csi))) - s_old_old);


                  v_new = (4.0/3.)*v_old - (1.0/3.)*v_old_old + (4.0/3.)*dt*RhsV -(2.0/3.)*dt*RhsV_old;
                  Rsystem.solution -> set(dof_indices2[v_var],v_new);
                  Rsystem.get_vector("v_prev").set(dof_indices2[0],v_old);
                  w_new = (4.0/3.)*w_old - (1.0/3.)*w_old_old + (4.0/3.)*dt*RhsW -(2.0/3.)*dt*RhsW_old;
                  Rsystem.get_vector("w_prev").set(dof_indices2[0],w_old);
                  Rsystem.solution -> set(dof_indices2[w_var],w_new);
                  s_new = (4.0/3.)*s_old - (1.0/3.)*s_old_old + (4.0/3.)*dt*RhsS -(2.0/3.)*dt*RhsS_old;
                  Rsystem.get_vector("s_prev").set(dof_indices2[0],s_old);
                  Rsystem.solution -> set(dof_indices2[s_var],s_new);
                }

            }
            else{

                  double RhsV = ( ((1.0 - HofX(u_old,u_c))*(1.0 - v_old))/(tau_v2_minus*HofX(u_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old,u_v)))) - ((HofX(u_old,u_c)*v_old)/(tau_v_plus));
                  double RhsW = ( ((1.0 - HofX(u_old,u_c))*(1.0 - w_old))/(tau_w2_minus*HofX(u_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old,u_w)))) - ((HofX(u_old,u_c)*w_old)/(tau_w_plus));
                  double RhsS = (r_s_plus*(HofX(u_old,u_c)) + r_s_minus*(1.0 - HofX(u_old,u_c))) * (.5*(1.0 + tanh(k*(u_old - u_csi))) - s_old);

                  v_new = v_old + dt*RhsV;
                  Rsystem.solution -> set(dof_indices2[v_var],v_new);
                  w_new = w_old + dt*RhsW;
                  Rsystem.solution -> set(dof_indices2[w_var],w_new);
                  s_new = s_old + dt*RhsS;
                  Rsystem.solution -> set(dof_indices2[s_var],s_new);
            }

            double freact = 0.0;
            double Ifival = (-v_old*HofX(u_old,u_c)*(u_old-u_c)*(u_m-u_old)) / (tau_d);
            double Isival = (-w_old*s_old) / (tau_si);
            double Isoval = (((u_old-u_o)*(1.0 - HofX(u_old,u_so))) / tau_o) + HofX(u_old,u_so)*tau_a + .5*(a_so-tau_a)*(1.0 + tanh((u_old - b_so)/(c_so)));

            double Istim = 0.0;
            double Istim2 = 0.0;

            if(SpiralBool == 0){
                if( zDim == 0. ){
                    if(time < IstimD && time > stimulus_start_time && y > stimulus_miny && y < stimulus_maxy && x > stimulus_minx && x < stimulus_maxx){
                      Istim = IstimV;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                      //libMesh::out << x << std::endl;
                    }
                }
                else{
                    if(time < IstimD && time > stimulus_start_time && y > stimulus_miny && y < stimulus_maxy && x > stimulus_minx && x < stimulus_maxx && z > stimulus_minz && z < stimulus_maxz){
                      Istim = IstimV;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                      //libMesh::out << x << std::endl;
                    }
                }
                if(convergence_test){
                  freact = -kcubic*((u_old - u0)*(u_old - u1)*(u_old - u2));// - 1*r_new*(u_old-u0)) - Istim  - Istim2;
                }
                else{
                    if(StimPlace.compare(0,13,"Transmembrane") == 0){
                        freact = (-(Ifival + Isival + Isoval)) - Istim  - Istim2;
                    }
                    else{
                        freact = (-(Ifival + Isival + Isoval));
                    }
                }
            }

            else if(SpiralBool == 1){
                if(time < IstimD && x < 0. && y > 0.){
                    Istim = IstimV;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                //libMesh::out << Istim << std::endl;
                }
                freact = (-(Ifival + Isival + Isoval)) - Istim  - Istim2;
            }

            else if(SpiralBool == 2){
                if(time < IstimD &&  x < tissue_minx + 0.5 && y > 0. ){
                    Istim = IstimV*1.;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                    //libMesh::out << Istim << std::endl;
                }
                if(time < IstimD &&  x > tissue_maxx - 0.15 ){
                    Istim = IstimV*1.;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                    //libMesh::out << Istim << std::endl;
                }
                //SECOND STIMULUS
                //if(time > 150. && time < 152. &&  x > tissue_maxx - 0.15 && y < 0. ){
                    //Istim = IstimV*.5;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                    //libMesh::out << Istim << std::endl;
                //}

                freact = (-(Ifival + Isival + Isoval)) - Istim  - Istim2;
            }




            
            //libMesh::out << freact << "    " << x << "    " << dof_indices[0] << std::endl;
            //libMesh::out << x << std::endl;


            Bsystem.get_vector("In").set(dof_indices[u_var], freact);           



        }

    }
    Bsystem.get_vector("In").close();
    Bsystem.get_vector("Inm1").close();
  //Rsystem.get_vector("Rvalues") = Rnew;
  

  }




  if(integrator.compare(0,4,"SBDF") != 0){

    TransientLinearImplicitSystem &Bsystem = es.get_system <TransientLinearImplicitSystem> ("parabolic");
    Bsystem.update();
    const unsigned int u_var = Bsystem.variable_number ("V");
    const DofMap & dof_map = Bsystem.get_dof_map();


    for(const auto & node : mesh.local_node_ptr_range()){

        dof_map.dof_indices (node, dof_indices);
        dof_map2.dof_indices (node, dof_indices2);

        const Real x = (*node)(0);
        const Real y = (*node)(1);
        const Real z = (*node)(2);

        if(dof_indices2.size() > 0){

          if(time == dt){
            v_old = 1.;
            w_old = 1.;
            s_old = 0.;
            u_old = (*Bsystem.current_local_solution)  (dof_indices[u_var]);

            if(SpiralBool == 1){
                if( y < 0){
                    v_old = 0.;
                    w_old = 0.;
                    s_old = 1.;
                    u_old = 1.;
                }
            }

            else if(SpiralBool == 2){

                double a = .5;
                double k = .09;
                int thickness = 15;
                int finalDim = thickness*1000; //15,000
                double bathRegion = xDim/2.0 - tissue_maxx;
                double factorAxis = 3.0*(xDim/2.-bathRegion);
                phi[0] = 0.;
                x1[0] = -a*exp(k*phi[0])*cos(phi[0]);
                y1[0] = -a*exp(k*phi[0])*sin(phi[0]);
                for(int i = 1; i < 1000; i++){
                    phi[i] = phi[i-1] + ((factorAxis*acos(-1) - phi[0])/1000.);
                    x1[i] = -a*exp(k*phi[i])*cos(phi[i]);
                    y1[i] = -a*exp(k*phi[i])*sin(phi[i]);
                }
                for(int i = 0; i < 1000; i++){
                    if( std::abs(x - x1[i]) <= threshSpiral && std::abs(y - y1[i]) <= threshSpiral){
                        //conditions in notes seen from paraview
                        /*
                         * initial conditions for multiple spirals to happen:
                            u - spiral of 1. and rest of 0
                            v - inverse of spiral of 1. and rest of 0
                            w - spiral of .8 and rest of 1.
                            s - same spiral of .1 and rest of 0
                        */
                        v_old = 0.;
                        w_old = 0.;
                        s_old = 1.;
                        u_old = 1.;
                        //libMesh::out << "HERE --> x = " << x << ";     y = " << y << ";    with threshold of the spiral: " << threshSpiral << std::endl;
                    }
                }
            }

          }
          else{
            v_old = (*Rsystem.current_local_solution) (dof_indices2[v_var]);
            w_old = (*Rsystem.current_local_solution) (dof_indices2[w_var]);        
            s_old = (*Rsystem.current_local_solution) (dof_indices2[s_var]);        
            u_old = (*Bsystem.current_local_solution)  (dof_indices[u_var]);
          }

          double RhsV = ( ((1.0 - HofX(u_old,u_c))*(1.0 - v_old))/(tau_v2_minus*HofX(u_old,u_v) + tau_v1_minus*(1.0 - HofX(u_old,u_v)))) - ((HofX(u_old,u_c)*v_old)/(tau_v_plus));
          double RhsW = ( ((1.0 - HofX(u_old,u_c))*(1.0 - w_old))/(tau_w2_minus*HofX(u_old,u_w) + tau_w1_minus*(1.0 - HofX(u_old,u_w)))) - ((HofX(u_old,u_c)*w_old)/(tau_w_plus));
          double RhsS = (r_s_plus*(HofX(u_old,u_c)) + r_s_minus*(1.0 - HofX(u_old,u_c))) * (.5*(1.0 + tanh(k*(u_old - u_csi))) - s_old);

          v_new = v_old + dt*RhsV;
          Rsystem.solution -> set(dof_indices2[v_var],v_new);
          w_new = w_old + dt*RhsW;
          Rsystem.solution -> set(dof_indices2[w_var],w_new);
          s_new = s_old + dt*RhsS;
          Rsystem.solution -> set(dof_indices2[s_var],s_new);
    

          double freact = 0.0;
          double Ifival = (-v_old*HofX(u_old,u_c)*(u_old-u_c)*(u_m-u_old)) / (tau_d);
          double Isival = (-w_old*s_old) / (tau_si);
          double Isoval = (((u_old-u_o)*(1.0 - HofX(u_old,u_so))) / tau_o) + HofX(u_old,u_so)*tau_a + .5*(a_so-tau_a)*(1.0 + tanh((u_old - b_so)/(c_so)));

          double Istim = 0.0;
          double Istim2 = 0.0;

          if(SpiralBool == 0){
                if( zDim == 0. ){
                    if(time < IstimD && time > stimulus_start_time && y > stimulus_miny && y < stimulus_maxy && x > stimulus_minx && x < stimulus_maxx){
                      Istim = IstimV;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                      //libMesh::out << x << std::endl;
                    }
                }
                else{
                    if(time < IstimD && time > stimulus_start_time && y > stimulus_miny && y < stimulus_maxy && x > stimulus_minx && x < stimulus_maxx && z > stimulus_minz && z < stimulus_maxz){
                      Istim = IstimV;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                      //libMesh::out << x << std::endl;
                    }
                }
                if(convergence_test){
                  //freact = -kcubic*((u_old - u0)*(u_old - u1)*(u_old - u2));// - 1*r_new*(u_old-u0)) - Istim  - Istim2;
                }
                else{
                    if(StimPlace.compare(0,13,"Transmembrane") == 0){
                        freact = (-(Ifival + Isival + Isoval)) - Istim  - Istim2;
                    }
                    else{
                        freact = (-(Ifival + Isival + Isoval));
                    }
                }
            }

            else if(SpiralBool == 1){
                if(time < IstimD && x < 0. && y > 0.){
                    Istim = IstimV;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                //libMesh::out << Istim << std::endl;
                }
                freact = (-(Ifival + Isival + Isoval)) - Istim  - Istim2;
            }

            else if(SpiralBool == 2){
                if(time < IstimD &&  x < tissue_minx + 0.5 && y > 0. ){
                    Istim = IstimV*1.;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                    //libMesh::out << Istim << std::endl;
                }
                if(time < IstimD &&  x > tissue_maxx - 0.15 ){
                    Istim = IstimV*1.;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                    //libMesh::out << Istim << std::endl;
                }
                //SECOND STIMULUS
                //if(time > 150. && time < 152. &&  x > tissue_maxx - 0.15 && y < 0. ){
                    //Istim = IstimV*.5;//with TIME < 1.2, -.08 is the minimum current stimulus to initiate AP propagation
                    //libMesh::out << Istim << std::endl;
                //}

                freact = (-(Ifival + Isival + Isoval)) - Istim  - Istim2;
            }

          //libMesh::out << freact << "    " << x << "    " << dof_indices[0] << std::endl;
          //libMesh::out << x << std::endl;


          Bsystem.get_vector("In").set(dof_indices[u_var], freact);



        }



    }
    Bsystem.get_vector("In").close();
    //Rsystem.get_vector("Rvalues") = Rnew;
  
    
  }



}








void init_cd_exact (EquationSystems & es, const double xDim)
{
    TransientLinearImplicitSystem & Bsystem = es.get_system <TransientLinearImplicitSystem> ("bidomain");
    //LinearImplicitSystem & Bsystem2 = es.get_system <LinearImplicitSystem> ("BidomainCheck");
    //LinearImplicitSystem & Psystem = es.get_system <LinearImplicitSystem> ("Parabolic");
    Bsystem.update();
    //Psystem.update();

    MeshBase & mesh = es.get_mesh();


      auto femSolu = es.get_system("bidomain").variable_number("V");
      auto femSolu2 = es.get_system("bidomain").variable_number("Ve");



      const unsigned int dim = mesh.mesh_dimension();
      const DofMap & dof_map = es.get_system("bidomain").get_dof_map();
      const DofMap & dof_map2 = es.get_system("Recovery").get_dof_map();

      FEType fe_type = dof_map.variable_type(femSolu);
      std::unique_ptr<FEBase> fe (FEBase::build(dim, fe_type));

      QGauss qrule (dim, TENTH);

        // Tell the finite element object to use our quadrature rule.
      fe->attach_quadrature_rule (&qrule);

      const std::vector<Point> & q_point = fe->get_xyz();
      const std::vector<Real> & JxW = fe->get_JxW();
      const std::vector<std::vector<Real>> & phi = fe->get_phi();

      int rowsn = 0;
      int colsn = 0;

      double u_h, u_exact, ue_h, ue_exact;

      std::vector<dof_id_type> dof_indices;
      std::vector<dof_id_type> dof_indices2;

        for(const auto & node : mesh.local_node_ptr_range()){

                  dof_map.dof_indices (node, dof_indices);
                  dof_map2.dof_indices (node, dof_indices2);

                  const Real x = (*node)(0);
                  const Real y = (*node)(1);

                  if(dof_indices2.size() > 0){

                    u_exact = exact_solutionV_all(x, y, 2, 0.0, 0.0, xDim);
                    Bsystem.solution -> set(dof_indices[femSolu],u_exact);
                    ue_exact = exact_solutionV_all(x, y, 3, 0.0, 0.0, xDim);
                    Bsystem.solution -> set(dof_indices[femSolu2],ue_exact);

                  }
                  else{
                    ue_exact = exact_solutionV_all(x, y, 4, 0.0, 0.0, xDim);
                    Bsystem.solution -> set(dof_indices[femSolu2],ue_exact);
                  }



                }

              Bsystem.solution -> close();
      

      



}


void ForcingTermConvergence (libMesh::EquationSystems &es, const double dt, const double Beta, const double Cm, const double SigmaSI, const double SigmaSE, const double SigmaBath, const double CurTime, const double xDim, const double v0, const double v1, const double v2, const double kcubic )
{
  // Ignore unused parameter warnings when !LIBMESH_ENABLE_AMR.
  //libmesh_ignore(es, system_name);


  // It is a good idea to make sure we are assembling
  // the proper system.
  //libmesh_assert_equal_to (system_name, "bidomain");

  // Get a constant reference to the mesh object.
  const MeshBase & mesh = es.get_mesh();

  // The dimension that we are running
  const unsigned int dim = mesh.mesh_dimension();

  // Get a reference to the Convection-Diffusion system object.
  //LinearImplicitSystem & Esystem = es.get_system<LinearImplicitSystem> ("Elliptic");
  TransientLinearImplicitSystem & Bsystem = es.get_system <TransientLinearImplicitSystem> ("bidomain");

  //ExplicitSystem & system2 = es.get_system<ExplicitSystem> ("Restitution");

  //TransientLinearImplicitSystem & system3 = es.get_system<TransientLinearImplicitSystem> ("MembraneUstar");

  // Get a constant reference to the Finite Element type
  // for the first (and only) variable in the system.
  FEType fe_type_V = Bsystem.variable_type(0);
  //FEType fe_type_Ve = Bsystem.variable_type(1);
  //FEType fe_type2 = Psystem.variable_type(0);
  // Build a Finite Element object of the specified type.  Since the
  // FEBase::build() member dynamically creates memory we will
  // store the object as a std::unique_ptr<FEBase>.  This can be thought
  // of as a pointer that will clean up after itself.
  std::unique_ptr<FEBase> fe      (FEBase::build(dim, fe_type_V));
  //std::unique_ptr<FEBase> fe2      (FEBase::build(dim, fe_type_Ve));
  std::unique_ptr<FEBase> fe_face (FEBase::build(dim, fe_type_V));

  // A Gauss quadrature rule for numerical integration.
  // Let the FEType object decide what order rule is appropriate.
  QGauss qrule (dim,  TENTH);
  //QGauss qrule2 (dim,   fe_type2.default_quadrature_order());
  QGauss qface (dim-1, TENTH);

  // Tell the finite element object to use our quadrature rule.
  fe->attach_quadrature_rule      (&qrule);
  //fe2->attach_quadrature_rule      (&qrule);
  fe_face->attach_quadrature_rule (&qface);

  // Here we define some references to cell-specific data that
  // will be used to assemble the linear system.  We will start
  // with the element Jacobian * quadrature weight at each integration point.
  const std::vector<Real> & JxW      = fe->get_JxW();
  //const std::vector<Real> & JxW2      = fe2->get_JxW();
  const std::vector<Real> & JxW_face = fe_face->get_JxW();

  // The element shape functions evaluated at the quadrature points.
  const std::vector<std::vector<Real>> & phi = fe->get_phi();
  //const std::vector<std::vector<Real>> & phi2 = fe2->get_phi();
  const std::vector<std::vector<Real>> & psi = fe_face->get_phi();

  // The element shape function gradients evaluated at the quadrature
  // points.
  const std::vector<std::vector<RealGradient>> & dphi = fe->get_dphi();
  //const std::vector<std::vector<RealGradient>> & dphi2 = fe2->get_dphi();

  // The XY locations of the quadrature points used for face integration
  const std::vector<Point> & q_point = fe->get_xyz();
  const std::vector<Point> & qface_points = fe_face->get_xyz();

  // A reference to the DofMap object for this system.  The DofMap
  // object handles the index translation from node and element numbers
  // to degree of freedom numbers.  We will talk more about the DofMap
  // in future examples.
  const DofMap & dof_map = Bsystem.get_dof_map();
  //const DofMap & dof_map2 = Bsystem.get_dof_map();
  //const DofMap & dof_map2 = system2.get_dof_map();

  // Define data structures to contain the element matrix
  // and right-hand-side vector contribution.  Following
  // basic finite element terminology we will denote these
  // "Ke" and "Fe".

  DenseVector<Number> Fe;



  DenseSubVector<Number>
    FV(Fe),
    FVe(Fe);


  //DenseMatrix<Number> Rloc;

  // This vector will hold the degree of freedom indices for
  // the element.  These define where in the global system
  // the element degrees of freedom get mapped.
  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_V;
  std::vector<dof_id_type> dof_indices_Ve;
  //std::vector<dof_id_type> dof_indices2;

  // Here we extract the velocity & parameters that we put in the
  // EquationSystems object.

  //const Real dt = es.parameters.get<Real>   ("dt");

  //SparseMatrix<Number> & matrix = Bsystem.get_system_matrix();

  //SparseMatrix<Number> & matrix = Esystem.get_system_matrix();

  // Now we will loop over all the elements in the mesh that
  // live on the local processor. We will compute the element
  // matrix and right-hand-side contribution.  Since the mesh
  // will be refined we want to only consider the ACTIVE elements,
  // hence we use a variant of the active_elem_iterator.
  /*for(const auto & node : mesh.local_node_ptr_range()){


    dof_map.dof_indices (node, dof_indices);

    if (elem->subdomain_id()==1){
      dof_map2.dof_indices (node, dof_indices2);
    }


  }
  */
  //libMesh::out << "  BEFORE LOOPING " << std::endl;

  for (const auto & elem : mesh.active_local_element_ptr_range())
    {
      // Get the degree of freedom indices for the
      // current element.  These define where in the global
      // matrix and right-hand-side this element will
      // contribute to.
      dof_map.dof_indices (elem, dof_indices);
      dof_map.dof_indices (elem, dof_indices_V, 0);
      dof_map.dof_indices (elem, dof_indices_Ve, 1);

      const unsigned int n_dofs   = dof_indices.size();
      const unsigned int n_V_dofs = dof_indices_V.size();
      const unsigned int n_Ve_dofs = dof_indices_Ve.size();

      //dof_map2.dof_indices (elem, dof_indices2);

      // Compute the element-specific data for the current
      // element.  This involves computing the location of the
      // quadrature points (q_point) and the shape functions
      // (phi, dphi) for the current element.
      fe->reinit (elem);

      // Zero the element matrix and right-hand side before
      // summing them.  We use the resize member here because
      // the number of degrees of freedom might have changed from
      // the last element.  Note that this will be the case if the
      // element type is different (i.e. the last element was a
      // triangle, now we are on a quadrilateral).


        //fe2->reinit (elem);




        Fe.resize (n_dofs);




      //Rloc.resize (dof_indices.size(), dof_indices.size());
      //libMesh::out << Rloc.size() << std::endl;

      //Rloc = es.parameters.get<DenseMatrix <Number> >("rlocal");



          //THIS IS NOW DEPENDENT ON FF,SS,NN

          //sigma(0,0) = 1.;
          //sigma(1,1) = 1.;
          //sigma(2,2) = 1.;

      //double Cm = InParam[6]; //uF/cm^2
      //double Beta = InParam[9]; //cm^-1

      //libMesh::out << Beta << " is the surface-volume ratio" << std::endl;

      // Now we will build the element matrix and right-hand-side.
      // Constructing the RHS requires the solution and its
      // gradient from the previous timestep.  This myst be
      // calculated at each quadrature point by summing the
      // solution degree-of-freedom values by the appropriate
      // weight functions.


      FV.reposition (0*n_V_dofs, n_V_dofs);
      FVe.reposition (1*n_V_dofs, n_Ve_dofs);

      //libMesh::out << "  BEFORE LOOPING QP " << std::endl;
      //if (elem->subdomain_id()==1){
        for (unsigned int qp=0; qp<qrule.n_points(); qp++)
        {
          // Values to hold the old solution & its gradient.

          //Gradient grad_u_old;


          //grad_u_old = es.get_system("Convection-Diffusion").point_value(femSolu, q_point[qp], elem);

          // Compute the old solution & its gradient.
          /*
          for (std::size_t l=0; l<phi.size(); l++)
            {
              //u_old += phi[l][qp]*Esystem.old_solution  (dof_indices[l]);
              //u_h = es.get_system("Convection-Diffusion").point_value(femSolu, q_point[qp], elem);
              //u_old += phi[l][qp]*Esystem.old_solution  (dof_indices[l]);


              // This will work,
              // grad_u_old += dphi[l][qp]*Esystem.old_solution (dof_indices[l]);
              // but we can do it without creating a temporary like this:
              //grad_u_old.add_scaled (dphi[l][qp], Esystem.old_solution (dof_indices[l]));
            }
            */

          //FeV
          //libMesh::out << "  BEFORE FV  " << std::endl;
          for (unsigned int i=0; i<n_V_dofs; i++){

            //FV(i) = JxW[qp]*(    (Beta*Cm/dt)*exact_solutionV(q_point[qp](0),q_point[qp](1), 2, Bsystem.time+dt, 0.0) + sigmaSI*exact_solutionV(q_point[qp](0),q_point[qp](1), 200, Bsystem.time+dt, 0.0) + sigmaSI*exact_solutionV(q_point[qp](0),q_point[qp](1), 300, Bsystem.time+dt, 0.0)      );
            

            FV(i) += JxW[qp]*((    CalculateF( q_point[qp](0), q_point[qp](1), 0, CurTime, 0.0, dt, Beta, Cm, SigmaSI, SigmaSE, SigmaBath, xDim, v0, v1, v2, kcubic)      )*(phi[i][qp]));
          
          
          }



          //FeVe
          //libMesh::out << "  BEFORE FVE  " << std::endl;
          for (unsigned int i=0; i<n_Ve_dofs; i++){

            if(elem->subdomain_id()==1){

              FVe(i) += JxW[qp]*((     CalculateF( q_point[qp](0), q_point[qp](1), 1, CurTime, 0.0, dt, Beta, Cm, SigmaSI, SigmaSE, SigmaBath, xDim, v0, v1, v2, kcubic)      )*(phi[i][qp]));
            
            }

            else{

              FVe(i) += JxW[qp]*((     CalculateF( q_point[qp](0), q_point[qp](1), 2, CurTime, 0.0, dt, Beta, Cm, SigmaSI, SigmaSE, SigmaBath, xDim, v0, v1, v2, kcubic)     )*(phi[i][qp]));
            
            }

          }



        }

        {

              // The penalty value.
          //libMesh::out << "  BEFORE ADDING BC  " << std::endl;
                          for (auto s : elem->side_index_range()){

                            //libMesh::out << elem << std::endl;
                                        if (elem->neighbor_ptr(s) == nullptr)
                                          {
                                          //libMesh::out << "SOMETHING        " << s << std::endl;

                                            std::unique_ptr<const Elem> side (elem->build_side_ptr(s,false));
                                            //libMesh::out << side.get() << std::endl;
                                            // Loop over the nodes on the side.
                                            for (auto ns : side->node_index_range())
                                              {
                                                // The location on the boundary of the current
                                                // node.
                                              //libMesh::out << ns << std::endl;

                                                const Real xf = side->point(ns)(0);
                                                const Real yf = side->point(ns)(1);

                                                // The penalty value.  \f$ \frac{1}{\epsilon \f$
                                                const Real penalty = 1.e10;

                                                // The boundary values.

                                                // Set v = 0 everywhere
                                                double Ve_value;
                                                Ve_value = exact_solutionV_all( xf, yf, 4, CurTime, 0.0, xDim);
                                                
                                                
                                                // Find the node on the element matching this node on
                                                // the side.  That defined where in the element matrix
                                                // the boundary condition will be applied.
                                                //libMesh::out << "  BEFORE CONDITIONS " << std::endl;
                                                
                                                for (auto n : elem->node_index_range()){
                                                  if (elem->node_id(n) == side->node_id(ns) && s == 1 || s == 3)
                                                    {
                                                      // Matrix contribution.
                                                      //KVeVe(n,n) += penalty;
                                                      //Kvv(n,n) += penalty;

                                                      // Right-hand-side contribution.
                                                      FVe(n) += penalty*Ve_value;
                                                      //Fv(n) += penalty*v_value;
                                                    }
                                                  }
                                                
                                                //libMesh::out << "  AFTER CONDITIONS " << std::endl;

                                              } // end face node loop
                                          }
                          }// end if (elem->neighbor(side) == nullptr)

                }


      // At this point the interior element integration has
      // been completed.  However, we have not yet addressed
      // boundary conditions.  For this example we will only
      // consider simple Dirichlet boundary conditions imposed
      // via the penalty method.
      //
      // The following loops over the sides of the element.
      // If the element has no neighbor on a side then that
      // side MUST live on a boundary of the domain.
        //libMesh::out << "  BEFORE ADDING VECTOR  " << std::endl;

        Bsystem.get_vector("ForcingConv").add_vector    (Fe, dof_indices);

        //libMesh::out << "  AFTER ADDING VECTOR  " << std::endl;

}

  //libMesh::out << "  AFTER LOOPING     " << Bsystem.rhs << std::endl;

      Bsystem.get_vector("ForcingConv").close();

      //libMesh::out << "  AFTER CLOSE " << std::endl;



}




