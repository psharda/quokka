//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef GAUSS_HPP
#define GAUSS_HPP

#include <array>
#include <cmath>
#include <limits>

#include "AMReX.H"

namespace quokka::math::quadrature
{
namespace detail
{

template <class T> struct gauss_constant_category {
	static const unsigned value =
	    (std::numeric_limits<T>::is_specialized == 0) ? 999
	    : (std::numeric_limits<T>::radix == 2)
		? ((std::numeric_limits<T>::digits <= std::numeric_limits<float>::digits) && std::is_convertible<float, T>::value		? 0
		   : (std::numeric_limits<T>::digits <= std::numeric_limits<double>::digits) && std::is_convertible<double, T>::value		? 1
		   : (std::numeric_limits<T>::digits <= std::numeric_limits<long double>::digits) && std::is_convertible<long double, T>::value ? 2
		   :
#ifdef BOOST_HAS_FLOAT128
		   (std::numeric_limits<T>::digits <= 113) && std::is_constructible<__float128, T>::value ? 3
		   :
#endif
		   (std::numeric_limits<T>::digits10 <= 110) ? 4
							     : 999)
	    : (std::numeric_limits<T>::digits10 <= 110) ? 4
							: 999;
};

template <class Real, unsigned N, unsigned Category> class gauss_detail;

template <class T> class gauss_detail<T, 7, 0>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 4> const &
	{
		static constexpr std::array<T, 4> data = {
		    0.000000000e+00F,
		    4.058451514e-01F,
		    7.415311856e-01F,
		    9.491079123e-01F,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 4> const &
	{
		static constexpr std::array<T, 4> data = {
		    4.179591837e-01F,
		    3.818300505e-01F,
		    2.797053915e-01F,
		    1.294849662e-01F,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 7, 1>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 4> const &
	{
		static constexpr std::array<T, 4> data = {
		    0.00000000000000000e+00,
		    4.05845151377397167e-01,
		    7.41531185599394440e-01,
		    9.49107912342758525e-01,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 4> const &
	{
		static constexpr std::array<T, 4> data = {
		    4.17959183673469388e-01,
		    3.81830050505118945e-01,
		    2.79705391489276668e-01,
		    1.29484966168869693e-01,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 7, 2>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 4> const &
	{
		static constexpr std::array<T, 4> data = {
		    0.00000000000000000000000000000000000e+00L,
		    4.05845151377397166906606412076961463e-01L,
		    7.41531185599394439863864773280788407e-01L,
		    9.49107912342758524526189684047851262e-01L,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 4> const &
	{
		static constexpr std::array<T, 4> data = {
		    4.17959183673469387755102040816326531e-01L,
		    3.81830050505118944950369775488975134e-01L,
		    2.79705391489276667901467771423779582e-01L,
		    1.29484966168869693270611432679082018e-01L,
		};
		return data;
	}
};
#ifdef BOOST_HAS_FLOAT128
template <class T> class gauss_detail<T, 7, 3>
{
      public:
	static std::array<T, 4> const &abscissa()
	{
		static const std::array<T, 4> data = {
		    0.00000000000000000000000000000000000e+00Q,
		    4.05845151377397166906606412076961463e-01Q,
		    7.41531185599394439863864773280788407e-01Q,
		    9.49107912342758524526189684047851262e-01Q,
		};
		return data;
	}
	static std::array<T, 4> const &weights()
	{
		static const std::array<T, 4> data = {
		    4.17959183673469387755102040816326531e-01Q,
		    3.81830050505118944950369775488975134e-01Q,
		    2.79705391489276667901467771423779582e-01Q,
		    1.29484966168869693270611432679082018e-01Q,
		};
		return data;
	}
};
#endif

template <class T> class gauss_detail<T, 10, 0>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 5> const &
	{
		static constexpr std::array<T, 5> data = {
		    1.488743390e-01F, 4.333953941e-01F, 6.794095683e-01F, 8.650633667e-01F, 9.739065285e-01F,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 5> const &
	{
		static constexpr std::array<T, 5> data = {
		    2.955242247e-01F, 2.692667193e-01F, 2.190863625e-01F, 1.494513492e-01F, 6.667134431e-02F,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 10, 1>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 5> const &
	{
		static constexpr std::array<T, 5> data = {
		    1.48874338981631211e-01, 4.33395394129247191e-01, 6.79409568299024406e-01, 8.65063366688984511e-01, 9.73906528517171720e-01,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 5> const &
	{
		static constexpr std::array<T, 5> data = {
		    2.95524224714752870e-01, 2.69266719309996355e-01, 2.19086362515982044e-01, 1.49451349150580593e-01, 6.66713443086881376e-02,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 10, 2>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 5> const &
	{
		static constexpr std::array<T, 5> data = {
		    1.48874338981631210884826001129719985e-01L, 4.33395394129247190799265943165784162e-01L, 6.79409568299024406234327365114873576e-01L,
		    8.65063366688984510732096688423493049e-01L, 9.73906528517171720077964012084452053e-01L,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 5> const &
	{
		static constexpr std::array<T, 5> data = {
		    2.95524224714752870173892994651338329e-01L, 2.69266719309996355091226921569469353e-01L, 2.19086362515982043995534934228163192e-01L,
		    1.49451349150580593145776339657697332e-01L, 6.66713443086881375935688098933317929e-02L,
		};
		return data;
	}
};
#ifdef BOOST_HAS_FLOAT128
template <class T> class gauss_detail<T, 10, 3>
{
      public:
	static std::array<T, 5> const &abscissa()
	{
		static const std::array<T, 5> data = {
		    1.48874338981631210884826001129719985e-01Q, 4.33395394129247190799265943165784162e-01Q, 6.79409568299024406234327365114873576e-01Q,
		    8.65063366688984510732096688423493049e-01Q, 9.73906528517171720077964012084452053e-01Q,
		};
		return data;
	}
	static std::array<T, 5> const &weights()
	{
		static const std::array<T, 5> data = {
		    2.95524224714752870173892994651338329e-01Q, 2.69266719309996355091226921569469353e-01Q, 2.19086362515982043995534934228163192e-01Q,
		    1.49451349150580593145776339657697332e-01Q, 6.66713443086881375935688098933317929e-02Q,
		};
		return data;
	}
};
#endif

template <class T> class gauss_detail<T, 15, 0>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 8> const &
	{
		static constexpr std::array<T, 8> data = {
		    0.000000000e+00F, 2.011940940e-01F, 3.941513471e-01F, 5.709721726e-01F,
		    7.244177314e-01F, 8.482065834e-01F, 9.372733924e-01F, 9.879925180e-01F,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 8> const &
	{
		static constexpr std::array<T, 8> data = {
		    2.025782419e-01F, 1.984314853e-01F, 1.861610000e-01F, 1.662692058e-01F,
		    1.395706779e-01F, 1.071592205e-01F, 7.036604749e-02F, 3.075324200e-02F,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 15, 1>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 8> const &
	{
		static constexpr std::array<T, 8> data = {
		    0.00000000000000000e+00, 2.01194093997434522e-01, 3.94151347077563370e-01, 5.70972172608538848e-01,
		    7.24417731360170047e-01, 8.48206583410427216e-01, 9.37273392400705904e-01, 9.87992518020485428e-01,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 8> const &
	{
		static constexpr std::array<T, 8> data = {
		    2.02578241925561273e-01, 1.98431485327111576e-01, 1.86161000015562211e-01, 1.66269205816993934e-01,
		    1.39570677926154314e-01, 1.07159220467171935e-01, 7.03660474881081247e-02, 3.07532419961172684e-02,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 15, 2>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 8> const &
	{
		static constexpr std::array<T, 8> data = {
		    0.00000000000000000000000000000000000e+00L, 2.01194093997434522300628303394596208e-01L, 3.94151347077563369897207370981045468e-01L,
		    5.70972172608538847537226737253910641e-01L, 7.24417731360170047416186054613938010e-01L, 8.48206583410427216200648320774216851e-01L,
		    9.37273392400705904307758947710209471e-01L, 9.87992518020485428489565718586612581e-01L,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 8> const &
	{
		static constexpr std::array<T, 8> data = {
		    2.02578241925561272880620199967519315e-01L, 1.98431485327111576456118326443839325e-01L, 1.86161000015562211026800561866422825e-01L,
		    1.66269205816993933553200860481208811e-01L, 1.39570677926154314447804794511028323e-01L, 1.07159220467171935011869546685869303e-01L,
		    7.03660474881081247092674164506673385e-02L, 3.07532419961172683546283935772044177e-02L,
		};
		return data;
	}
};
#ifdef BOOST_HAS_FLOAT128
template <class T> class gauss_detail<T, 15, 3>
{
      public:
	static std::array<T, 8> const &abscissa()
	{
		static const std::array<T, 8> data = {
		    0.00000000000000000000000000000000000e+00Q, 2.01194093997434522300628303394596208e-01Q, 3.94151347077563369897207370981045468e-01Q,
		    5.70972172608538847537226737253910641e-01Q, 7.24417731360170047416186054613938010e-01Q, 8.48206583410427216200648320774216851e-01Q,
		    9.37273392400705904307758947710209471e-01Q, 9.87992518020485428489565718586612581e-01Q,
		};
		return data;
	}
	static std::array<T, 8> const &weights()
	{
		static const std::array<T, 8> data = {
		    2.02578241925561272880620199967519315e-01Q, 1.98431485327111576456118326443839325e-01Q, 1.86161000015562211026800561866422825e-01Q,
		    1.66269205816993933553200860481208811e-01Q, 1.39570677926154314447804794511028323e-01Q, 1.07159220467171935011869546685869303e-01Q,
		    7.03660474881081247092674164506673385e-02Q, 3.07532419961172683546283935772044177e-02Q,
		};
		return data;
	}
};
#endif

template <class T> class gauss_detail<T, 20, 0>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 10> const &
	{
		static constexpr std::array<T, 10> data = {
		    7.652652113e-02F, 2.277858511e-01F, 3.737060887e-01F, 5.108670020e-01F, 6.360536807e-01F,
		    7.463319065e-01F, 8.391169718e-01F, 9.122344283e-01F, 9.639719273e-01F, 9.931285992e-01F,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 10> const &
	{
		static constexpr std::array<T, 10> data = {
		    1.527533871e-01F, 1.491729865e-01F, 1.420961093e-01F, 1.316886384e-01F, 1.181945320e-01F,
		    1.019301198e-01F, 8.327674158e-02F, 6.267204833e-02F, 4.060142980e-02F, 1.761400714e-02F,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 20, 1>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 10> const &
	{
		static constexpr std::array<T, 10> data = {
		    7.65265211334973338e-02, 2.27785851141645078e-01, 3.73706088715419561e-01, 5.10867001950827098e-01, 6.36053680726515025e-01,
		    7.46331906460150793e-01, 8.39116971822218823e-01, 9.12234428251325906e-01, 9.63971927277913791e-01, 9.93128599185094925e-01,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 10> const &
	{
		static constexpr std::array<T, 10> data = {
		    1.52753387130725851e-01, 1.49172986472603747e-01, 1.42096109318382051e-01, 1.31688638449176627e-01, 1.18194531961518417e-01,
		    1.01930119817240435e-01, 8.32767415767047487e-02, 6.26720483341090636e-02, 4.06014298003869413e-02, 1.76140071391521183e-02,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 20, 2>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 10> const &
	{
		static constexpr std::array<T, 10> data = {
		    7.65265211334973337546404093988382110e-02L, 2.27785851141645078080496195368574625e-01L, 3.73706088715419560672548177024927237e-01L,
		    5.10867001950827098004364050955250998e-01L, 6.36053680726515025452836696226285937e-01L, 7.46331906460150792614305070355641590e-01L,
		    8.39116971822218823394529061701520685e-01L, 9.12234428251325905867752441203298113e-01L, 9.63971927277913791267666131197277222e-01L,
		    9.93128599185094924786122388471320278e-01L,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 10> const &
	{
		static constexpr std::array<T, 10> data = {
		    1.52753387130725850698084331955097593e-01L, 1.49172986472603746787828737001969437e-01L, 1.42096109318382051329298325067164933e-01L,
		    1.31688638449176626898494499748163135e-01L, 1.18194531961518417312377377711382287e-01L, 1.01930119817240435036750135480349876e-01L,
		    8.32767415767047487247581432220462061e-02L, 6.26720483341090635695065351870416064e-02L, 4.06014298003869413310399522749321099e-02L,
		    1.76140071391521183118619623518528164e-02L,
		};
		return data;
	}
};
#ifdef BOOST_HAS_FLOAT128
template <class T> class gauss_detail<T, 20, 3>
{
      public:
	static std::array<T, 10> const &abscissa()
	{
		static const std::array<T, 10> data = {
		    7.65265211334973337546404093988382110e-02Q, 2.27785851141645078080496195368574625e-01Q, 3.73706088715419560672548177024927237e-01Q,
		    5.10867001950827098004364050955250998e-01Q, 6.36053680726515025452836696226285937e-01Q, 7.46331906460150792614305070355641590e-01Q,
		    8.39116971822218823394529061701520685e-01Q, 9.12234428251325905867752441203298113e-01Q, 9.63971927277913791267666131197277222e-01Q,
		    9.93128599185094924786122388471320278e-01Q,
		};
		return data;
	}
	static std::array<T, 10> const &weights()
	{
		static const std::array<T, 10> data = {
		    1.52753387130725850698084331955097593e-01Q, 1.49172986472603746787828737001969437e-01Q, 1.42096109318382051329298325067164933e-01Q,
		    1.31688638449176626898494499748163135e-01Q, 1.18194531961518417312377377711382287e-01Q, 1.01930119817240435036750135480349876e-01Q,
		    8.32767415767047487247581432220462061e-02Q, 6.26720483341090635695065351870416064e-02Q, 4.06014298003869413310399522749321099e-02Q,
		    1.76140071391521183118619623518528164e-02Q,
		};
		return data;
	}
};
#endif

template <class T> class gauss_detail<T, 25, 0>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 13> const &
	{
		static constexpr std::array<T, 13> data = {
		    0.000000000e+00F, 1.228646926e-01F, 2.438668837e-01F, 3.611723058e-01F, 4.730027314e-01F, 5.776629302e-01F, 6.735663685e-01F,
		    7.592592630e-01F, 8.334426288e-01F, 8.949919979e-01F, 9.429745712e-01F, 9.766639215e-01F, 9.955569698e-01F,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 13> const &
	{
		static constexpr std::array<T, 13> data = {
		    1.231760537e-01F, 1.222424430e-01F, 1.194557635e-01F, 1.148582591e-01F, 1.085196245e-01F, 1.005359491e-01F, 9.102826198e-02F,
		    8.014070034e-02F, 6.803833381e-02F, 5.490469598e-02F, 4.093915670e-02F, 2.635498662e-02F, 1.139379850e-02F,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 25, 1>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 13> const &
	{
		static constexpr std::array<T, 13> data = {
		    0.00000000000000000e+00, 1.22864692610710396e-01, 2.43866883720988432e-01, 3.61172305809387838e-01, 4.73002731445714961e-01,
		    5.77662930241222968e-01, 6.73566368473468364e-01, 7.59259263037357631e-01, 8.33442628760834001e-01, 8.94991997878275369e-01,
		    9.42974571228974339e-01, 9.76663921459517511e-01, 9.95556969790498098e-01,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 13> const &
	{
		static constexpr std::array<T, 13> data = {
		    1.23176053726715451e-01, 1.22242442990310042e-01, 1.19455763535784772e-01, 1.14858259145711648e-01, 1.08519624474263653e-01,
		    1.00535949067050644e-01, 9.10282619829636498e-02, 8.01407003350010180e-02, 6.80383338123569172e-02, 5.49046959758351919e-02,
		    4.09391567013063127e-02, 2.63549866150321373e-02, 1.13937985010262879e-02,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 25, 2>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 13> const &
	{
		static constexpr std::array<T, 13> data = {
		    0.00000000000000000000000000000000000e+00L, 1.22864692610710396387359818808036806e-01L, 2.43866883720988432045190362797451586e-01L,
		    3.61172305809387837735821730127640667e-01L, 4.73002731445714960522182115009192041e-01L, 5.77662930241222967723689841612654067e-01L,
		    6.73566368473468364485120633247622176e-01L, 7.59259263037357630577282865204360976e-01L, 8.33442628760834001421021108693569569e-01L,
		    8.94991997878275368851042006782804954e-01L, 9.42974571228974339414011169658470532e-01L, 9.76663921459517511498315386479594068e-01L,
		    9.95556969790498097908784946893901617e-01L,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 13> const &
	{
		static constexpr std::array<T, 13> data = {
		    1.23176053726715451203902873079050142e-01L, 1.22242442990310041688959518945851506e-01L, 1.19455763535784772228178126512901047e-01L,
		    1.14858259145711648339325545869555809e-01L, 1.08519624474263653116093957050116619e-01L, 1.00535949067050644202206890392685827e-01L,
		    9.10282619829636498114972207028916534e-02L, 8.01407003350010180132349596691113023e-02L, 6.80383338123569172071871856567079686e-02L,
		    5.49046959758351919259368915404733242e-02L, 4.09391567013063126556234877116459537e-02L, 2.63549866150321372619018152952991449e-02L,
		    1.13937985010262879479029641132347736e-02L,
		};
		return data;
	}
};
#ifdef BOOST_HAS_FLOAT128
template <class T> class gauss_detail<T, 25, 3>
{
      public:
	static std::array<T, 13> const &abscissa()
	{
		static const std::array<T, 13> data = {
		    0.00000000000000000000000000000000000e+00Q, 1.22864692610710396387359818808036806e-01Q, 2.43866883720988432045190362797451586e-01Q,
		    3.61172305809387837735821730127640667e-01Q, 4.73002731445714960522182115009192041e-01Q, 5.77662930241222967723689841612654067e-01Q,
		    6.73566368473468364485120633247622176e-01Q, 7.59259263037357630577282865204360976e-01Q, 8.33442628760834001421021108693569569e-01Q,
		    8.94991997878275368851042006782804954e-01Q, 9.42974571228974339414011169658470532e-01Q, 9.76663921459517511498315386479594068e-01Q,
		    9.95556969790498097908784946893901617e-01Q,
		};
		return data;
	}
	static std::array<T, 13> const &weights()
	{
		static const std::array<T, 13> data = {
		    1.23176053726715451203902873079050142e-01Q, 1.22242442990310041688959518945851506e-01Q, 1.19455763535784772228178126512901047e-01Q,
		    1.14858259145711648339325545869555809e-01Q, 1.08519624474263653116093957050116619e-01Q, 1.00535949067050644202206890392685827e-01Q,
		    9.10282619829636498114972207028916534e-02Q, 8.01407003350010180132349596691113023e-02Q, 6.80383338123569172071871856567079686e-02Q,
		    5.49046959758351919259368915404733242e-02Q, 4.09391567013063126556234877116459537e-02Q, 2.63549866150321372619018152952991449e-02Q,
		    1.13937985010262879479029641132347736e-02Q,
		};
		return data;
	}
};
#endif

template <class T> class gauss_detail<T, 30, 0>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 15> const &
	{
		static constexpr std::array<T, 15> data = {
		    5.147184256e-02F, 1.538699136e-01F, 2.546369262e-01F, 3.527047255e-01F, 4.470337695e-01F,
		    5.366241481e-01F, 6.205261830e-01F, 6.978504948e-01F, 7.677774321e-01F, 8.295657624e-01F,
		    8.825605358e-01F, 9.262000474e-01F, 9.600218650e-01F, 9.836681233e-01F, 9.968934841e-01F,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 15> const &
	{
		static constexpr std::array<T, 15> data = {
		    1.028526529e-01F, 1.017623897e-01F, 9.959342059e-02F, 9.636873717e-02F, 9.212252224e-02F,
		    8.689978720e-02F, 8.075589523e-02F, 7.375597474e-02F, 6.597422988e-02F, 5.749315622e-02F,
		    4.840267283e-02F, 3.879919257e-02F, 2.878470788e-02F, 1.846646831e-02F, 7.968192496e-03F,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 30, 1>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 15> const &
	{
		static constexpr std::array<T, 15> data = {
		    5.14718425553176958e-02, 1.53869913608583547e-01, 2.54636926167889846e-01, 3.52704725530878113e-01, 4.47033769538089177e-01,
		    5.36624148142019899e-01, 6.20526182989242861e-01, 6.97850494793315797e-01, 7.67777432104826195e-01, 8.29565762382768397e-01,
		    8.82560535792052682e-01, 9.26200047429274326e-01, 9.60021864968307512e-01, 9.83668123279747210e-01, 9.96893484074649540e-01,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 15> const &
	{
		static constexpr std::array<T, 15> data = {
		    1.02852652893558840e-01, 1.01762389748405505e-01, 9.95934205867952671e-02, 9.63687371746442596e-02, 9.21225222377861287e-02,
		    8.68997872010829798e-02, 8.07558952294202154e-02, 7.37559747377052063e-02, 6.59742298821804951e-02, 5.74931562176190665e-02,
		    4.84026728305940529e-02, 3.87991925696270496e-02, 2.87847078833233693e-02, 1.84664683110909591e-02, 7.96819249616660562e-03,
		};
		return data;
	}
};

template <class T> class gauss_detail<T, 30, 2>
{
      public:
	AMREX_GPU_DEVICE static auto abscissa() -> std::array<T, 15> const &
	{
		static constexpr std::array<T, 15> data = {
		    5.14718425553176958330252131667225737e-02L, 1.53869913608583546963794672743255920e-01L, 2.54636926167889846439805129817805108e-01L,
		    3.52704725530878113471037207089373861e-01L, 4.47033769538089176780609900322854000e-01L, 5.36624148142019899264169793311072794e-01L,
		    6.20526182989242861140477556431189299e-01L, 6.97850494793315796932292388026640068e-01L, 7.67777432104826194917977340974503132e-01L,
		    8.29565762382768397442898119732501916e-01L, 8.82560535792052681543116462530225590e-01L, 9.26200047429274325879324277080474004e-01L,
		    9.60021864968307512216871025581797663e-01L, 9.83668123279747209970032581605662802e-01L, 9.96893484074649540271630050918695283e-01L,
		};
		return data;
	}
	AMREX_GPU_DEVICE static auto weights() -> std::array<T, 15> const &
	{
		static constexpr std::array<T, 15> data = {
		    1.02852652893558840341285636705415044e-01L, 1.01762389748405504596428952168554045e-01L, 9.95934205867952670627802821035694765e-02L,
		    9.63687371746442596394686263518098651e-02L, 9.21225222377861287176327070876187672e-02L, 8.68997872010829798023875307151257026e-02L,
		    8.07558952294202153546949384605297309e-02L, 7.37559747377052062682438500221907342e-02L, 6.59742298821804951281285151159623612e-02L,
		    5.74931562176190664817216894020561288e-02L, 4.84026728305940529029381404228075178e-02L, 3.87991925696270495968019364463476920e-02L,
		    2.87847078833233693497191796112920436e-02L, 1.84664683110909591423021319120472691e-02L, 7.96819249616660561546588347467362245e-03L,
		};
		return data;
	}
};
#ifdef BOOST_HAS_FLOAT128
template <class T> class gauss_detail<T, 30, 3>
{
      public:
	static std::array<T, 15> const &abscissa()
	{
		static const std::array<T, 15> data = {
		    5.14718425553176958330252131667225737e-02Q, 1.53869913608583546963794672743255920e-01Q, 2.54636926167889846439805129817805108e-01Q,
		    3.52704725530878113471037207089373861e-01Q, 4.47033769538089176780609900322854000e-01Q, 5.36624148142019899264169793311072794e-01Q,
		    6.20526182989242861140477556431189299e-01Q, 6.97850494793315796932292388026640068e-01Q, 7.67777432104826194917977340974503132e-01Q,
		    8.29565762382768397442898119732501916e-01Q, 8.82560535792052681543116462530225590e-01Q, 9.26200047429274325879324277080474004e-01Q,
		    9.60021864968307512216871025581797663e-01Q, 9.83668123279747209970032581605662802e-01Q, 9.96893484074649540271630050918695283e-01Q,
		};
		return data;
	}
	static std::array<T, 15> const &weights()
	{
		static const std::array<T, 15> data = {
		    1.02852652893558840341285636705415044e-01Q, 1.01762389748405504596428952168554045e-01Q, 9.95934205867952670627802821035694765e-02Q,
		    9.63687371746442596394686263518098651e-02Q, 9.21225222377861287176327070876187672e-02Q, 8.68997872010829798023875307151257026e-02Q,
		    8.07558952294202153546949384605297309e-02Q, 7.37559747377052062682438500221907342e-02Q, 6.59742298821804951281285151159623612e-02Q,
		    5.74931562176190664817216894020561288e-02Q, 4.84026728305940529029381404228075178e-02Q, 3.87991925696270495968019364463476920e-02Q,
		    2.87847078833233693497191796112920436e-02Q, 1.84664683110909591423021319120472691e-02Q, 7.96819249616660561546588347467362245e-03Q,
		};
		return data;
	}
};
#endif

} // namespace detail

template <class Real, unsigned N> class gauss : public detail::gauss_detail<Real, N, detail::gauss_constant_category<Real>::value>
{
	using base = detail::gauss_detail<Real, N, detail::gauss_constant_category<Real>::value>;

      public:
	template <class F> AMREX_GPU_DEVICE static auto integrate(F f, Real *pL1 = nullptr) -> decltype(f(Real(0.0)))
	{
		// In many math texts, K represents the field of real or complex numbers.
		// Too bad we can't put blackboard bold into C++ source!
		using K = decltype(f(Real(0)));
		static_assert(!std::is_integral<K>::value, "The return type cannot be integral, it must be either a real or complex floating point type.");
		using std::abs;
		unsigned non_zero_start = 1;
		K result = Real(0);
		if (N & 1) {
			result = f(Real(0)) * base::weights()[0];
		} else {
			result = 0;
			non_zero_start = 0;
		}
		Real L1 = abs(result);
		for (unsigned i = non_zero_start; i < base::abscissa().size(); ++i) {
			K fp = f(base::abscissa()[i]);
			K fm = f(-base::abscissa()[i]);
			result += (fp + fm) * base::weights()[i];
			L1 += (abs(fp) + abs(fm)) * base::weights()[i];
		}
		if (pL1) {
			*pL1 = L1;
		}
		return result;
	}

	template <class F> AMREX_GPU_DEVICE static auto integrate(F f, Real a, Real b, Real *pL1 = nullptr) -> decltype(f(Real(0.0)))
	{
		using K = decltype(f(a));
		if (!(std::isnan)(a) && !(std::isnan)(b)) {
			// Infinite limits:
			Real min_inf = -std::numeric_limits<Real>::max();
			if ((a <= min_inf) && (b >= std::numeric_limits<Real>::max())) {
				auto u = [&](const Real &t) -> K {
					Real t_sq = t * t;
					Real inv = 1 / (1 - t_sq);
					K res = f(t * inv) * (1 + t_sq) * inv * inv;
					return res;
				};
				return integrate(u, pL1);
			}

			// Right limit is infinite:
			if ((std::isfinite(a)) && (b >= std::numeric_limits<Real>::max())) {
				auto u = [&](const Real &t) -> K {
					Real z = 1 / (t + 1);
					Real arg = 2 * z + a - 1;
					K res = f(arg) * z * z;
					return res;
				};
				K Q = Real(2) * integrate(u, pL1);
				if (pL1) {
					*pL1 *= 2;
				}
				return Q;
			}

			if ((std::isfinite(b)) && (a <= -std::numeric_limits<Real>::max())) {
				auto v = [&](const Real &t) -> K {
					Real z = 1 / (t + 1);
					Real arg = 2 * z - 1;
					K res = f(b - arg) * z * z;
					return res;
				};
				K Q = Real(2) * integrate(v, pL1);
				if (pL1) {
					*pL1 *= 2;
				}
				return Q;
			}

			if ((std::isfinite(a)) && (std::isfinite(b))) {
				if (a == b) {
					return K(0);
				}
				if (b < a) {
					return -integrate(f, b, a, pL1);
				}
				Real avg = 0.5 * (a + b);
				Real scale = 0.5 * (b - a);

				auto u = [&](Real z) -> K { return f(avg + scale * z); };
				K Q = scale * integrate(u, pL1);

				if (pL1) {
					*pL1 *= scale;
				}
				return Q;
			}
		}
		return std::numeric_limits<K>::signaling_NaN();
	}
};

} // namespace quokka::math::quadrature

#endif // GAUSS_HPP
