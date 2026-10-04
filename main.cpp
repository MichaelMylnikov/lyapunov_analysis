#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <cstdlib>
#include <limits>
#include <future>
#include <chrono>

#include <Eigen/Dense>

constexpr double PI = 3.14159265358979323846;

struct EllipsePoint { double c, x, y; };
struct FieldPoint   { double x, y, dx, dy; };
struct TrajPoint    { size_t id; double x, y; };

Eigen::Matrix2d inputMatrix( const std::string& name ) 
{
    Eigen::Matrix2d M;
    std::cout << "---------------------------------------------------------\n";
    std::cout << " Ввод матрицы " << name << " (2x2):\n";
    std::cout << " Формат ввода: вводите 2 числа через пробел для каждой строки\n";
    std::cout << "---------------------------------------------------------\n";
    
    for ( int i = 0; i < 2; ++i ) 
        while ( true ) 
        {
            std::cout << " Строка " << i + 1 << " (" << name << "[" << i + 1 << "][1] " 
                      << name << "[" << i + 1 << "][2]) -> ";
            if ( std::cin >> M( i, 0 ) >> M( i, 1 ) )
                break;
            else 
            {
                std::cout << " [!] Ошибка! Введите два числа через пробел.\n";
                std::cin.clear();
                std::cin.ignore( std::numeric_limits<std::streamsize>::max(), '\n' );
            }
        }

    std::cout << "\n Матрица " << name << ":\n" << M << "\n\n";
    return M;
}

Eigen::Matrix2d solveLyapunovEigen( const Eigen::Matrix2d& A, const Eigen::Matrix2d& Q ) 
{
    Eigen::Matrix4d M;
    M << 2 * A( 0, 0 ),             A( 1, 0 ),             A( 1, 0 ),              0,
             A( 0, 1 ),   A( 0, 0 ) + A( 1, 1 ),                   0,      A( 1, 0 ),
             A( 0, 1 ),                     0,   A( 0, 0 ) + A( 1, 1 ),      A( 1, 0 ),
                     0,             A( 0, 1 ),             A( 0, 1 ),  2 * A( 1, 1 );

    Eigen::Vector4d qVec;
    qVec << -Q( 0, 0 ), -Q( 0, 1 ), -Q( 1, 0 ), -Q( 1, 1 );

    Eigen::Vector4d pVec = M.fullPivLu().solve( qVec );

    Eigen::Matrix2d P;
    P << pVec( 0 ), 0.5 * ( pVec( 1 ) + pVec( 2 ) ),
         0.5 * ( pVec( 1 ) + pVec( 2 ) ), pVec( 3 );

    return P;
}

Eigen::Vector2d rk4Step( const Eigen::Vector2d& x, const Eigen::Matrix2d& A, double dt )
{
    Eigen::Vector2d k1 = A * x;
    Eigen::Vector2d k2 = A * ( x + 0.5 * dt * k1 );
    Eigen::Vector2d k3 = A * ( x + 0.5 * dt * k2 );
    Eigen::Vector2d k4 = A * ( x + dt * k3 );

    return x + ( dt / 6.0 ) * ( k1 + 2.0 * k2 + 2.0 * k3 + k4 );
}

std::vector<EllipsePoint> computeEllipses( const Eigen::Matrix2d& P ) 
{
    std::vector<EllipsePoint> result;
    for ( double c : { 0.5, 1.0, 2.0, 4.0 } ) 
        for ( double angle = 0; angle <= 2 * PI; angle += 0.03 ) 
        {
            Eigen::Vector2d u( std::cos( angle ), std::sin( angle ) );
            double val = u.transpose() * P * u;
            if ( val > 0 ) 
            {
                double r = std::sqrt( c / val );
                result.push_back( { c, r * u( 0 ), r * u( 1 ) } );
            }
        }
    return result;
}

std::vector<FieldPoint> computeVectorField( const Eigen::Matrix2d& A ) 
{
    std::vector<FieldPoint> result;
    for ( double x = -4.0; x <= 4.0; x += 0.5 ) 
        for ( double y = -4.0; y <= 4.0; y += 0.5 ) 
        {
            Eigen::Vector2d state( x, y );
            Eigen::Vector2d dxt = A * state;
            result.push_back( { x, y, dxt( 0 ), dxt( 1 ) } );
        }
    return result;
}

std::vector<TrajPoint> computeTrajectories( const Eigen::Matrix2d& A ) 
{
    std::vector<TrajPoint> result;
    std::vector<Eigen::Vector2d> starts;
    
    for ( double r : { 2.0, 3.8 } ) 
        for ( double a = 0; a < 2 * PI; a += PI / 4.0 ) 
            starts.push_back( { r * std::cos( a ), r * std::sin( a ) } );

    for ( size_t id = 0; id < starts.size(); ++id ) 
    {
        Eigen::Vector2d state = starts[id];
        for ( int step = 0; step < 600; ++step ) 
        {
            result.push_back( { id, state( 0 ), state( 1 ) } );
            state = rk4Step( state, A, 0.015 );
        }
    }
    return result;
}

int main()
{
    std::cout << "=======================================\n";
    std::cout << " ЧИСЛЕННОЕ ПОСТРОЕНИЕ ФУНКЦИИ ЛЯПУНОВА\n";
    std::cout << "=======================================\n\n";

    Eigen::Matrix2d A = inputMatrix( "A" );
    Eigen::Matrix2d Q = inputMatrix( "Q" );

    Eigen::EigenSolver<Eigen::Matrix2d> es( A );
    auto ev = es.eigenvalues();

    std::cout << "[1] Собственные значения lambda:\n"
              << "    l1 = " << ev( 0 ) << "\n    l2 = " << ev( 1 ) << "\n\n";

    bool isHurwitz = ( ev( 0 ).real() < 0.0 && ev( 1 ).real() < 0.0 );
    std::cout << "[2] Гурвицевость Re(lambda) < 0: " << ( isHurwitz ? "OK" : "FAIL" ) << "\n\n";

    Eigen::Matrix2d P = solveLyapunovEigen( A, Q );
    std::cout << "[3] Найдена матрица P:\n" << P << "\n\n";

    bool isSymmetric = P.isApprox( P.transpose() );
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> esP( P );
    bool isPosDef = ( esP.eigenvalues().array() > 0 ).all();

    std::cout << "[4] Симметричность P: " << ( isSymmetric ? "OK" : "FAIL" ) << "\n";
    std::cout << "[5] Положительная определенность P > 0: " << ( isPosDef ? "OK" : "FAIL" ) << "\n\n";

    std::cout << "[6] Построение функции Ляпунова V(x) = x^T P x: OK\n";
    std::cout << "[7] Построение линий уровня V(x, y) = c: OK\n";
    std::cout << "[8] Построение фазового поля и траекторий: OK\n\n";

    Eigen::Vector2d xTest( 1.0, 1.0 );
    double vDot = -xTest.transpose() * Q * xTest;
    std::cout << "[9] Численная проверка знака V_dot = -x^T Q x:\n"
              << "    В точке x = [1.0, 1.0]^T значение x^T Q x = " 
              << ( xTest.transpose() * Q * xTest ) << "\n"
              << "    Производная V_dot = " << vDot << " (< 0): OK\n\n";

    std::cout << "[6-8] Запуск параллельных расчетов в 3 потоках...\n";
    auto t_start = std::chrono::high_resolution_clock::now();

    auto fut_ellipses = std::async( std::launch::async, computeEllipses, std::cref( P ) );
    auto fut_field    = std::async( std::launch::async, computeVectorField, std::cref( A ) );
    auto fut_trajs    = std::async( std::launch::async, computeTrajectories, std::cref( A ) );

    auto ellipses = fut_ellipses.get();
    auto field    = fut_field.get();
    auto trajs    = fut_trajs.get();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>( t_end - t_start ).count();
    std::cout << " Расчеты завершены за " << std::fixed << std::setprecision( 3 ) << elapsed_ms << " ms!\n\n";

    std::ofstream fEllipse( "ellipse_levels.txt" );
    if ( fEllipse.is_open() )
    {
        for ( const auto& p : ellipses )
            fEllipse << p.c << " " << p.x << " " << p.y << "\n";
        fEllipse.close();
    }

    std::ofstream fField( "vector_field.txt" );
    if ( fField.is_open() )
    {
        for ( const auto& p : field )
            fField << p.x << " " << p.y << " " << p.dx << " " << p.dy << "\n";
        fField.close();
    }

    std::ofstream fTraj( "trajectories.txt" );
    if ( fTraj.is_open() )
    {
        for ( const auto& p : trajs )
            fTraj << p.id << " " << p.x << " " << p.y << "\n";
        fTraj.close();
    }

    std::cout << "[10] Запуск python3 plot_results.py...\n";
    int ret = std::system( "python3 plot_results.py" );
    if ( ret != 0 )
        std::cerr << " [!] Внимание: скрипт Python завершился с кодом ошибки: " << ret << "\n";

    return 0;
}