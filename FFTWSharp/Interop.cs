// The code in this file is provided courtesy of Tamas Szalay. Some functionality has been added.

// FFTWSharp
// ===========
// Basic C# wrapper for FFTW.
//
// Features
// ============
//    * Unmanaged function calls to main FFTW functions for both single and double precision
//    * Basic managed wrappers for FFTW plans and unmanaged arrays
//    * Test program that demonstrates basic functionality
//
// Notes
// ============
//    * Most of this was written in 2005
//    * Slightly updated since to get it running with Visual Studio Express 2010
//    * If you have a question about FFTW, ask the FFTW people, and not me. I did not write FFTW.
//    * If you have a question about this wrapper, probably still don't ask me, since I wrote it almost a decade ago.


using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Xml;

namespace FFTWSharp
{
    // Various Flags used by FFTW
    #region Enums
    /// <summary>
    /// FFTW planner flags
    /// </summary>
    [Flags]
    public enum fftw_flags : uint
    {
        /// <summary>
        /// Tells FFTW to find an optimized plan by actually computing several FFTs and measuring their execution time. 
        /// Depending on your machine, this can take some time (often a few seconds). Default (0x0). 
        /// </summary>
        Measure = 0,
        /// <summary>
        /// Specifies that an out-of-place transform is allowed to overwrite its 
        /// input array with arbitrary data; this can sometimes allow more efficient algorithms to be employed.
        /// </summary>
        DestroyInput = 1,
        /// <summary>
        /// Rarely used. Specifies that the algorithm may not impose any unusual alignment requirements on the input/output 
        /// arrays (i.e. no SIMD). This flag is normally not necessary, since the planner automatically detects 
        /// misaligned arrays. The only use for this flag is if you want to use the guru interface to execute a given 
        /// plan on a different array that may not be aligned like the original. 
        /// </summary>
        Unaligned = 2,
        /// <summary>
        /// Not used.
        /// </summary>
        ConserveMemory = 4,
        /// <summary>
        /// Like Patient, but considers an even wider range of algorithms, including many that we think are 
        /// unlikely to be fast, to produce the most optimal plan but with a substantially increased planning time. 
        /// </summary>
        Exhaustive = 8,
        /// <summary>
        /// Specifies that an out-of-place transform must not change its input array. 
        /// </summary>
        /// <remarks>
        /// This is ordinarily the default, 
        /// except for c2r and hc2r (i.e. complex-to-real) transforms for which DestroyInput is the default. 
        /// In the latter cases, passing PreserveInput will attempt to use algorithms that do not destroy the 
        /// input, at the expense of worse performance; for multi-dimensional c2r transforms, however, no 
        /// input-preserving algorithms are implemented and the planner will return null if one is requested.
        /// </remarks>
        PreserveInput = 16,
        /// <summary>
        /// Like Measure, but considers a wider range of algorithms and often produces a “more optimal?plan 
        /// (especially for large transforms), but at the expense of several times longer planning time 
        /// (especially for large transforms).
        /// </summary>
        Patient = 32,
        /// <summary>
        /// Specifies that, instead of actual measurements of different algorithms, a simple heuristic is 
        /// used to pick a (probably sub-optimal) plan quickly. With this flag, the input/output arrays 
        /// are not overwritten during planning. 
        /// </summary>
        Estimate = 64
    }

    /// <summary>
    /// Defines direction of operation
    /// </summary>
    public enum fftw_direction : int
    {
        /// <summary>
        /// Computes a regular DFT
        /// </summary>
        Forward = -1,
        /// <summary>
        /// Computes the inverse DFT
        /// </summary>
        Backward = 1
    }

    /// <summary>
    /// Kinds of real-to-real transforms
    /// </summary>
    public enum fftw_kind : uint
    {
        R2HC = 0,
        HC2R = 1,
        DHT = 2,
        REDFT00 = 3,
        REDFT01 = 4,
        REDFT10 = 5,
        REDFT11 = 6,
        RODFT00 = 7,
        RODFT01 = 8,
        RODFT10 = 9,
        RODFT11 = 10
    }
    #endregion

    // FFTW Interop Classes
    #region Single Precision
    /// <summary>
    /// Contains the Basic Interface FFTW functions for single-precision (float) operations
    /// </summary>
    public class fftwf
    {
        static fftwf()
        {
            FFTWPlatformsAdapter unsafeNativeMethods = new FFTWPlatformsAdapter("libfftw3f-3.dll");
        }
        /// <summary>
        /// Allocates FFTW-optimized unmanaged memory
        /// </summary>
        /// <param name="length">Amount to allocate, in bytes</param>
        /// <returns>Pointer to allocated memory</returns>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_malloc",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr malloc(int length);

        /// <summary>
        /// Deallocates memory allocated by FFTW malloc
        /// </summary>
        /// <param name="mem">Pointer to memory to release</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_free",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void free(IntPtr mem);

        /// <summary>
        /// Deallocates an FFTW plan and all associated resources
        /// </summary>
        /// <param name="plan">Pointer to the plan to release</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_destroy_plan",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void destroy_plan(IntPtr plan);

        /// <summary>
        /// Clears all memory used by FFTW, resets it to initial state. Does not replace destroy_plan and free
        /// </summary>
        /// <remarks>After calling fftw_cleanup, all existing plans become undefined, and you should not 
        /// attempt to execute them nor to destroy them. You can however create and execute/destroy new plans, 
        /// in which case FFTW starts accumulating wisdom information again. 
        /// fftw_cleanup does not deallocate your plans; you should still call fftw_destroy_plan for this purpose.</remarks>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_cleanup",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void cleanup();

        /// <summary>
        /// Sets the maximum time that can be used by the planner.
        /// </summary>
        /// <param name="seconds">Maximum time, in seconds.</param>
        /// <remarks>This function instructs FFTW to spend at most seconds seconds (approximately) in the planner. 
        /// If seconds == -1.0 (the default value), then planning time is unbounded. 
        /// Otherwise, FFTW plans with a progressively wider range of algorithms until the the given time limit is 
        /// reached or the given range of algorithms is explored, returning the best available plan. For example, 
        /// specifying fftw_flags.Patient first plans in Estimate mode, then in Measure mode, then finally (time 
        /// permitting) in Patient. If fftw_flags.Exhaustive is specified instead, the planner will further progress to 
        /// Exhaustive mode. 
        /// </remarks>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_set_timelimit",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void set_timelimit(double seconds);

        /// <summary>
        /// Executes an FFTW plan, provided that the input and output arrays still exist
        /// </summary>
        /// <param name="plan">Pointer to the plan to execute</param>
        /// <remarks>execute (and equivalents) is the only function in FFTW guaranteed to be thread-safe.</remarks>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_execute",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void execute(IntPtr plan);

        /// <summary>
        /// Creates a plan for a 1-dimensional complex-to-complex DFT
        /// </summary>
        /// <param name="n">The logical size of the transform</param>
        /// <param name="direction">Specifies the direction of the transform</param>
        /// <param name="input">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_1d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_1d(int n, IntPtr input, IntPtr output,
            fftw_direction direction, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 2-dimensional complex-to-complex DFT
        /// </summary>
        /// <param name="nx">The logical size of the transform along the first dimension</param>
        /// <param name="ny">The logical size of the transform along the second dimension</param>
        /// <param name="direction">Specifies the direction of the transform</param>
        /// <param name="input">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_2d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_2d(int nx, int ny, IntPtr input, IntPtr output,
            fftw_direction direction, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 3-dimensional complex-to-complex DFT
        /// </summary>
        /// <param name="nx">The logical size of the transform along the first dimension</param>
        /// <param name="ny">The logical size of the transform along the second dimension</param>
        /// <param name="nz">The logical size of the transform along the third dimension</param>
        /// <param name="direction">Specifies the direction of the transform</param>
        /// <param name="input">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_3d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_3d(int nx, int ny, int nz, IntPtr input, IntPtr output,
            fftw_direction direction, fftw_flags flags);

        /// <summary>
        /// Creates a plan for an n-dimensional complex-to-complex DFT
        /// </summary>
        /// <param name="rank">Number of dimensions</param>
        /// <param name="n">Array containing the logical size along each dimension</param>
        /// <param name="direction">Specifies the direction of the transform</param>
        /// <param name="input">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft(int rank, int[] n, IntPtr input, IntPtr output,
            fftw_direction direction, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 1-dimensional real-to-complex DFT
        /// </summary>
        /// <param name="n">Number of REAL (input) elements in the transform</param>
        /// <param name="input">Pointer to an array of 4-byte real numbers</param>
        /// <param name="output">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_r2c_1d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_r2c_1d(int n, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 2-dimensional real-to-complex DFT
        /// </summary>
        /// <param name="nx">Number of REAL (input) elements in the transform along the first dimension</param>
        /// <param name="ny">Number of REAL (input) elements in the transform along the second dimension</param>
        /// <param name="input">Pointer to an array of 4-byte real numbers</param>
        /// <param name="output">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_r2c_2d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_r2c_2d(int nx, int ny, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 3-dimensional real-to-complex DFT
        /// </summary>
        /// <param name="nx">Number of REAL (input) elements in the transform along the first dimension</param>
        /// <param name="ny">Number of REAL (input) elements in the transform along the second dimension</param>
        /// <param name="nz">Number of REAL (input) elements in the transform along the third dimension</param>
        /// <param name="input">Pointer to an array of 4-byte real numbers</param>
        /// <param name="output">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_r2c_3d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_r2c_3d(int nx, int ny, int nz, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for an n-dimensional real-to-complex DFT
        /// </summary>
        /// <param name="rank">Number of dimensions</param>
        /// <param name="n">Array containing the number of REAL (input) elements along each dimension</param>
        /// <param name="input">Pointer to an array of 4-byte real numbers</param>
        /// <param name="output">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_r2c",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_r2c(int rank, int[] n, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 1-dimensional complex-to-real DFT
        /// </summary>
        /// <param name="n">Number of REAL (output) elements in the transform</param>
        /// <param name="input">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 4-byte real numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_c2r_1d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_c2r_1d(int n, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 2-dimensional complex-to-real DFT
        /// </summary>
        /// <param name="nx">Number of REAL (output) elements in the transform along the first dimension</param>
        /// <param name="ny">Number of REAL (output) elements in the transform along the second dimension</param>
        /// <param name="input">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 4-byte real numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_c2r_2d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_c2r_2d(int nx, int ny, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 3-dimensional complex-to-real DFT
        /// </summary>
        /// <param name="nx">Number of REAL (output) elements in the transform along the first dimension</param>
        /// <param name="ny">Number of REAL (output) elements in the transform along the second dimension</param>
        /// <param name="nz">Number of REAL (output) elements in the transform along the third dimension</param>
        /// <param name="input">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 4-byte real numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_c2r_3d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_c2r_3d(int nx, int ny, int nz, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for an n-dimensional complex-to-real DFT
        /// </summary>
        /// <param name="rank">Number of dimensions</param>
        /// <param name="n">Array containing the number of REAL (output) elements along each dimension</param>
        /// <param name="input">Pointer to an array of 8-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 4-byte real numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_dft_c2r",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_c2r(int rank, int[] n, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 1-dimensional real-to-real DFT
        /// </summary>
        /// <param name="n">Number of elements in the transform</param>
        /// <param name="input">Pointer to an array of 4-byte real numbers</param>
        /// <param name="output">Pointer to an array of 4-byte real numbers</param>
        /// <param name="kind">The kind of real-to-real transform to compute</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_r2r_1d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr r2r_1d(int n, IntPtr input, IntPtr output, fftw_kind kind, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 2-dimensional real-to-real DFT
        /// </summary>
        /// <param name="nx">Number of elements in the transform along the first dimension</param>
        /// <param name="ny">Number of elements in the transform along the second dimension</param>
        /// <param name="input">Pointer to an array of 4-byte real numbers</param>
        /// <param name="output">Pointer to an array of 4-byte real numbers</param>
        /// <param name="kindx">The kind of real-to-real transform to compute along the first dimension</param>
        /// <param name="kindy">The kind of real-to-real transform to compute along the second dimension</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_r2r_2d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr r2r_2d(int nx, int ny, IntPtr input, IntPtr output,
            fftw_kind kindx, fftw_kind kindy, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 3-dimensional real-to-real DFT
        /// </summary>
        /// <param name="nx">Number of elements in the transform along the first dimension</param>
        /// <param name="ny">Number of elements in the transform along the second dimension</param>
        /// <param name="nz">Number of elements in the transform along the third dimension</param>
        /// <param name="input">Pointer to an array of 4-byte real numbers</param>
        /// <param name="output">Pointer to an array of 4-byte real numbers</param>
        /// <param name="kindx">The kind of real-to-real transform to compute along the first dimension</param>
        /// <param name="kindy">The kind of real-to-real transform to compute along the second dimension</param>
        /// <param name="kindz">The kind of real-to-real transform to compute along the third dimension</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_r2r_3d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr r2r_3d(int nx, int ny, int nz, IntPtr input, IntPtr output,
            fftw_kind kindx, fftw_kind kindy, fftw_kind kindz, fftw_flags flags);

        /// <summary>
        /// Creates a plan for an n-dimensional real-to-real DFT
        /// </summary>
        /// <param name="rank">Number of dimensions</param>
        /// <param name="n">Array containing the number of elements in the transform along each dimension</param>
        /// <param name="input">Pointer to an array of 4-byte real numbers</param>
        /// <param name="output">Pointer to an array of 4-byte real numbers</param>
        /// <param name="kind">An array containing the kind of real-to-real transform to compute along each dimension</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_plan_r2r",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr r2r(int rank, int[] n, IntPtr input, IntPtr output,
            fftw_kind[] kind, fftw_flags flags);

        /// <summary>
        /// Returns (approximately) the number of flops used by a certain plan
        /// </summary>
        /// <param name="plan">The plan to measure</param>
        /// <param name="add">Reference to double to hold number of adds</param>
        /// <param name="mul">Reference to double to hold number of muls</param>
        /// <param name="fma">Reference to double to hold number of fmas (fused multiply-add)</param>
        /// <remarks>Total flops ~= add+mul+2*fma or add+mul+fma if fma is supported</remarks>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_flops",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void flops(IntPtr plan, ref double add, ref double mul, ref double fma);

        /// <summary>
        /// Outputs a "nerd-readable" version of the specified plan to stdout
        /// </summary>
        /// <param name="plan">The plan to output</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_print_plan",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void print_plan(IntPtr plan);

        /// <summary>
        /// Exports the accumulated Wisdom to the provided filename
        /// </summary>
        /// <param name="filename">The target filename</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_export_wisdom_to_filename",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void export_wisdom_to_filename(string filename);


        /// <summary>
        /// Imports Wisdom from provided filename
        /// </summary>
        /// <param name="filename">The filename to read from</param>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_import_wisdom_from_filename",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void import_wisdom_from_filename(string filename);

        /// <summary>
        /// Forgets the Wisdom
        /// </summary>
        [DllImport("libfftw3f-3.dll",
             EntryPoint = "fftwf_forget_wisdom",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void fftwf_forget_wisdom();
    }
    #endregion

    #region Double Precision
    /// <summary>
    /// Contains the Basic Interface FFTW functions for double-precision (double) operations
    /// </summary>
    public class fftw
    {
        static fftw()
        {
            FFTWPlatformsAdapter unsafeNativeMethods = new FFTWPlatformsAdapter("libfftw3-3.dll");
        }
        /// <summary>
        /// Allocates FFTW-optimized unmanaged memory
        /// </summary>
        /// <param name="length">Amount to allocate, in bytes</param>
        /// <returns>Pointer to allocated memory</returns>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_malloc",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr malloc(int length);

        /// <summary>
        /// Deallocates memory allocated by FFTW malloc
        /// </summary>
        /// <param name="mem">Pointer to memory to release</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_free",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void free(IntPtr mem);

        /// <summary>
        /// Deallocates an FFTW plan and all associated resources
        /// </summary>
        /// <param name="plan">Pointer to the plan to release</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_destroy_plan",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void destroy_plan(IntPtr plan);

        /// <summary>
        /// Clears all memory used by FFTW, resets it to initial state. Does not replace destroy_plan and free
        /// </summary>
        /// <remarks>After calling fftw_cleanup, all existing plans become undefined, and you should not 
        /// attempt to execute them nor to destroy them. You can however create and execute/destroy new plans, 
        /// in which case FFTW starts accumulating wisdom information again. 
        /// fftw_cleanup does not deallocate your plans; you should still call fftw_destroy_plan for this purpose.</remarks>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_cleanup",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void cleanup();

        /// <summary>
        /// Sets the maximum time that can be used by the planner.
        /// </summary>
        /// <param name="seconds">Maximum time, in seconds.</param>
        /// <remarks>This function instructs FFTW to spend at most seconds seconds (approximately) in the planner. 
        /// If seconds == -1.0 (the default value), then planning time is unbounded. 
        /// Otherwise, FFTW plans with a progressively wider range of algorithms until the the given time limit is 
        /// reached or the given range of algorithms is explored, returning the best available plan. For example, 
        /// specifying fftw_flags.Patient first plans in Estimate mode, then in Measure mode, then finally (time 
        /// permitting) in Patient. If fftw_flags.Exhaustive is specified instead, the planner will further progress to 
        /// Exhaustive mode. 
        /// </remarks>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_set_timelimit",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void set_timelimit(double seconds);

        /// <summary>
        /// Executes an FFTW plan, provided that the input and output arrays still exist
        /// </summary>
        /// <param name="plan">Pointer to the plan to execute</param>
        /// <remarks>execute (and equivalents) is the only function in FFTW guaranteed to be thread-safe.</remarks>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_execute",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void execute(IntPtr plan);

        /// <summary>
        /// Creates a plan for a 1-dimensional complex-to-complex DFT
        /// </summary>
        /// <param name="n">The logical size of the transform</param>
        /// <param name="direction">Specifies the direction of the transform</param>
        /// <param name="input">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_1d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_1d(int n, IntPtr input, IntPtr output,
            fftw_direction direction, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 2-dimensional complex-to-complex DFT
        /// </summary>
        /// <param name="nx">The logical size of the transform along the first dimension</param>
        /// <param name="ny">The logical size of the transform along the second dimension</param>
        /// <param name="direction">Specifies the direction of the transform</param>
        /// <param name="input">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_2d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_2d(int nx, int ny, IntPtr input, IntPtr output,
            fftw_direction direction, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 3-dimensional complex-to-complex DFT
        /// </summary>
        /// <param name="nx">The logical size of the transform along the first dimension</param>
        /// <param name="ny">The logical size of the transform along the second dimension</param>
        /// <param name="nz">The logical size of the transform along the third dimension</param>
        /// <param name="direction">Specifies the direction of the transform</param>
        /// <param name="input">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_3d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_3d(int nx, int ny, int nz, IntPtr input, IntPtr output,
            fftw_direction direction, fftw_flags flags);

        /// <summary>
        /// Creates a plan for an n-dimensional complex-to-complex DFT
        /// </summary>
        /// <param name="rank">Number of dimensions</param>
        /// <param name="n">Array containing the logical size along each dimension</param>
        /// <param name="direction">Specifies the direction of the transform</param>
        /// <param name="input">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft(int rank, int[] n, IntPtr input, IntPtr output,
            fftw_direction direction, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 1-dimensional real-to-complex DFT
        /// </summary>
        /// <param name="n">Number of REAL (input) elements in the transform</param>
        /// <param name="input">Pointer to an array of 8-byte real numbers</param>
        /// <param name="output">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_r2c_1d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_r2c_1d(int n, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 2-dimensional real-to-complex DFT
        /// </summary>
        /// <param name="nx">Number of REAL (input) elements in the transform along the first dimension</param>
        /// <param name="ny">Number of REAL (input) elements in the transform along the second dimension</param>
        /// <param name="input">Pointer to an array of 8-byte real numbers</param>
        /// <param name="output">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_r2c_2d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_r2c_2d(int nx, int ny, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 3-dimensional real-to-complex DFT
        /// </summary>
        /// <param name="nx">Number of REAL (input) elements in the transform along the first dimension</param>
        /// <param name="ny">Number of REAL (input) elements in the transform along the second dimension</param>
        /// <param name="nz">Number of REAL (input) elements in the transform along the third dimension</param>
        /// <param name="input">Pointer to an array of 8-byte real numbers</param>
        /// <param name="output">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_r2c_3d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_r2c_3d(int nx, int ny, int nz, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for an n-dimensional real-to-complex DFT
        /// </summary>
        /// <param name="rank">Number of dimensions</param>
        /// <param name="n">Array containing the number of REAL (input) elements along each dimension</param>
        /// <param name="input">Pointer to an array of 8-byte real numbers</param>
        /// <param name="output">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_r2c",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_r2c(int rank, int[] n, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 1-dimensional complex-to-real DFT
        /// </summary>
        /// <param name="n">Number of REAL (output) elements in the transform</param>
        /// <param name="input">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 8-byte real numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_c2r_1d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_c2r_1d(int n, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 2-dimensional complex-to-real DFT
        /// </summary>
        /// <param name="nx">Number of REAL (output) elements in the transform along the first dimension</param>
        /// <param name="ny">Number of REAL (output) elements in the transform along the second dimension</param>
        /// <param name="input">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 8-byte real numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_c2r_2d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_c2r_2d(int nx, int ny, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 3-dimensional complex-to-real DFT
        /// </summary>
        /// <param name="nx">Number of REAL (output) elements in the transform along the first dimension</param>
        /// <param name="ny">Number of REAL (output) elements in the transform along the second dimension</param>
        /// <param name="nz">Number of REAL (output) elements in the transform along the third dimension</param>
        /// <param name="input">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 8-byte real numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_c2r_3d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_c2r_3d(int nx, int ny, int nz, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for an n-dimensional complex-to-real DFT
        /// </summary>
        /// <param name="rank">Number of dimensions</param>
        /// <param name="n">Array containing the number of REAL (output) elements along each dimension</param>
        /// <param name="input">Pointer to an array of 16-byte complex numbers</param>
        /// <param name="output">Pointer to an array of 8-byte real numbers</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_dft_c2r",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr dft_c2r(int rank, int[] n, IntPtr input, IntPtr output, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 1-dimensional real-to-real DFT
        /// </summary>
        /// <param name="n">Number of elements in the transform</param>
        /// <param name="input">Pointer to an array of 8-byte real numbers</param>
        /// <param name="output">Pointer to an array of 8-byte real numbers</param>
        /// <param name="kind">The kind of real-to-real transform to compute</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_r2r_1d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr r2r_1d(int n, IntPtr input, IntPtr output, fftw_kind kind, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 2-dimensional real-to-real DFT
        /// </summary>
        /// <param name="nx">Number of elements in the transform along the first dimension</param>
        /// <param name="ny">Number of elements in the transform along the second dimension</param>
        /// <param name="input">Pointer to an array of 8-byte real numbers</param>
        /// <param name="output">Pointer to an array of 8-byte real numbers</param>
        /// <param name="kindx">The kind of real-to-real transform to compute along the first dimension</param>
        /// <param name="kindy">The kind of real-to-real transform to compute along the second dimension</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_r2r_2d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr r2r_2d(int nx, int ny, IntPtr input, IntPtr output,
            fftw_kind kindx, fftw_kind kindy, fftw_flags flags);

        /// <summary>
        /// Creates a plan for a 3-dimensional real-to-real DFT
        /// </summary>
        /// <param name="nx">Number of elements in the transform along the first dimension</param>
        /// <param name="ny">Number of elements in the transform along the second dimension</param>
        /// <param name="nz">Number of elements in the transform along the third dimension</param>
        /// <param name="input">Pointer to an array of 8-byte real numbers</param>
        /// <param name="output">Pointer to an array of 8-byte real numbers</param>
        /// <param name="kindx">The kind of real-to-real transform to compute along the first dimension</param>
        /// <param name="kindy">The kind of real-to-real transform to compute along the second dimension</param>
        /// <param name="kindz">The kind of real-to-real transform to compute along the third dimension</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_r2r_3d",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr r2r_3d(int nx, int ny, int nz, IntPtr input, IntPtr output,
            fftw_kind kindx, fftw_kind kindy, fftw_kind kindz, fftw_flags flags);

        /// <summary>
        /// Creates a plan for an n-dimensional real-to-real DFT
        /// </summary>
        /// <param name="rank">Number of dimensions</param>
        /// <param name="n">Array containing the number of elements in the transform along each dimension</param>
        /// <param name="input">Pointer to an array of 8-byte real numbers</param>
        /// <param name="output">Pointer to an array of 8-byte real numbers</param>
        /// <param name="kind">An array containing the kind of real-to-real transform to compute along each dimension</param>
        /// <param name="flags">Flags that specify the behavior of the planner</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_plan_r2r",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr r2r(int rank, int[] n, IntPtr input, IntPtr output,
            fftw_kind[] kind, fftw_flags flags);

        /// <summary>
        /// Returns (approximately) the number of flops used by a certain plan
        /// </summary>
        /// <param name="plan">The plan to measure</param>
        /// <param name="add">Reference to double to hold number of adds</param>
        /// <param name="mul">Reference to double to hold number of muls</param>
        /// <param name="fma">Reference to double to hold number of fmas (fused multiply-add)</param>
        /// <remarks>Total flops ~= add+mul+2*fma or add+mul+fma if fma is supported</remarks>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_flops",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void flops(IntPtr plan, ref double add, ref double mul, ref double fma);

        /// <summary>
        /// Outputs a "nerd-readable" version of the specified plan to stdout
        /// </summary>
        /// <param name="plan">The plan to output</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_print_plan",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void print_plan(IntPtr plan);

        /// <summary>
        /// Exports the accumulated Wisdom to the provided filename
        /// </summary>
        /// <param name="filename">The target filename</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_export_wisdom_to_filename",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void export_wisdom_to_filename(string filename);


        /// <summary>
        /// Imports Wisdom from provided filename
        /// </summary>
        /// <param name="filename">The filename to read from</param>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_import_wisdom_from_filename",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void import_wisdom_from_filename(string filename);

        /// <summary>
        /// Forgets the Wisdom
        /// </summary>
        [DllImport("libfftw3-3.dll",
             EntryPoint = "fftw_forget_wisdom",
             ExactSpelling = true,
             CallingConvention = CallingConvention.Cdecl)]
        public static extern void fftw_forget_wisdom();
    }
    #endregion
    /// <summary>
    /// copy 
    /// </summary>
    public class FFTWPlatformsAdapter
    {
        public FFTWPlatformsAdapter(string dllName)
        {
            fftw_DLL = dllName;
            FFTWPlatformsAdapter.DllFileExtension = ".dll";
            FFTWPlatformsAdapter.ConfigFileExtension = ".config";
            FFTWPlatformsAdapter.XmlConfigFileName = typeof(FFTWPlatformsAdapter).Namespace + FFTWPlatformsAdapter.DllFileExtension + FFTWPlatformsAdapter.ConfigFileExtension;
            FFTWPlatformsAdapter.staticSyncRoot = new object();
            FFTWPlatformsAdapter.PROCESSOR_ARCHITECTURE = "PROCESSOR_ARCHITECTURE";
            FFTWPlatformsAdapter._fftwNativeModuleFileName = null;
            FFTWPlatformsAdapter._fftwNativeModuleHandle = IntPtr.Zero;
            Initialize();
        }
        internal static string fftw_DLL = "libfftw3f-3.dll";

        /// <summary>
        /// The file extension used for dynamic link libraries.
        /// </summary>
        private static string DllFileExtension;

        /// <summary>
        /// The file extension used for the XML configuration file.
        /// </summary>
        private static string ConfigFileExtension;

        /// <summary>
        /// This is the name of the XML configuration file specific to the
        /// System.Data.fftw assembly.
        /// </summary>
        private static string XmlConfigFileName;

        /// <summary>
        /// This lock is used to protect the static _fftwNativeModuleFileName,
        /// _fftwNativeModuleHandle, and processorArchitecturePlatforms fields.
        /// </summary>
        private static object staticSyncRoot;

        /// <summary>
        /// This dictionary stores the mappings between processor architecture
        /// names and platform names.  These mappings are now used for two
        /// purposes.  First, they are used to determine if the assembly code
        /// base should be used instead of the location, based upon whether one
        /// or more of the named sub-directories exist within the assembly code
        /// base.  Second, they are used to assist in loading the appropriate
        /// fftw interop assembly into the current process.
        /// </summary>
        private static Dictionary<string, string> processorArchitecturePlatforms;

        /// <summary>
        /// The name of the environment variable containing the processor
        /// architecture of the current process.
        /// </summary>
        private static string PROCESSOR_ARCHITECTURE;

        /// <summary>
        /// The native module file name for the native fftw library or null.
        /// </summary>
        private static string _fftwNativeModuleFileName;

        /// <summary>
        /// The native module handle for the native fftw library or the value
        /// IntPtr.Zero.
        /// </summary>
        private static IntPtr _fftwNativeModuleHandle;
        /// Attempts to initialize this class by pre-loading the native fftw
		/// library for the processor architecture of the current process.
		/// </summary>
		internal static void Initialize()
        {
            //if (UnsafeNativeMethods.GetSettingValue("No_PreLoadfftw", null) != null)
            //{
            //    return;
            //}
            lock (FFTWPlatformsAdapter.staticSyncRoot)
            {
                if (FFTWPlatformsAdapter.processorArchitecturePlatforms == null)
                {
                    FFTWPlatformsAdapter.processorArchitecturePlatforms = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                    FFTWPlatformsAdapter.processorArchitecturePlatforms.Add("x86", "Win32");
                    FFTWPlatformsAdapter.processorArchitecturePlatforms.Add("AMD64", "x64");
                    FFTWPlatformsAdapter.processorArchitecturePlatforms.Add("IA64", "Itanium");
                    FFTWPlatformsAdapter.processorArchitecturePlatforms.Add("ARM", "WinCE");
                }
                if (FFTWPlatformsAdapter._fftwNativeModuleHandle == IntPtr.Zero)
                {
                    string baseDirectory = null;
                    string processorArchitecture = null;
                    FFTWPlatformsAdapter.SearchForDirectory(ref baseDirectory, ref processorArchitecture);
                    FFTWPlatformsAdapter.PreLoadfftwDll(baseDirectory, processorArchitecture, ref FFTWPlatformsAdapter._fftwNativeModuleFileName, ref FFTWPlatformsAdapter._fftwNativeModuleHandle);
                }
            }
        }

        /// <summary>
        /// Queries and returns the XML configuration file name for the assembly
        /// containing the managed System.Data.fftw components.
        /// </summary>
        /// <returns>
        /// The XML configuration file name -OR- null if it cannot be determined
        /// or does not exist.
        /// </returns>
        private static string GetXmlConfigFileName()
        {
            string path = AppDomain.CurrentDomain.BaseDirectory;
            string text = Path.Combine(path, FFTWPlatformsAdapter.XmlConfigFileName);
            if (File.Exists(text))
            {
                return text;
            }
            path = FFTWPlatformsAdapter.GetAssemblyDirectory();
            text = Path.Combine(path, FFTWPlatformsAdapter.XmlConfigFileName);
            if (File.Exists(text))
            {
                return text;
            }
            return null;
        }

        /// <summary>
        /// Queries and returns the value of the specified setting, using the XML
        /// configuration file and/or the environment variables for the current
        /// process and/or the current system, when available.
        /// </summary>
        /// <param name="name">
        /// The name of the setting.
        /// </param>
        /// <param name="default">
        /// The value to be returned if the setting has not been set explicitly
        /// or cannot be determined.
        /// </param>
        /// <returns>
        /// The value of the setting -OR- the default value specified by
        /// <paramref name="default" /> if it has not been set explicitly or
        /// cannot be determined.  By default, all references to existing
        /// environment variables will be expanded to their corresponding values
        /// within the value to be returned unless either the "No_Expand" or
        /// "No_Expand_<paramref name="name" />" environment variable is set [to
        /// anything].
        /// </returns>
        internal static string GetSettingValue(string name, string @default)
        {
            if (name == null)
            {
                return @default;
            }
            bool flag = true;
            if (Environment.GetEnvironmentVariable("No_Expand") != null)
            {
                flag = false;
            }
            else if (Environment.GetEnvironmentVariable(string.Format("No_Expand_{0}", name)) != null)
            {
                flag = false;
            }
            string text = Environment.GetEnvironmentVariable(name);
            if (flag && !string.IsNullOrEmpty(text))
            {
                text = Environment.ExpandEnvironmentVariables(text);
            }
            if (text != null)
            {
                return text;
            }
            try
            {
                string xmlConfigFileName = FFTWPlatformsAdapter.GetXmlConfigFileName();
                if (xmlConfigFileName == null)
                {
                    string result = @default;
                    return result;
                }
                XmlDocument xmlDocument = new XmlDocument();
                xmlDocument.Load(xmlConfigFileName);
                XmlElement xmlElement = xmlDocument.SelectSingleNode(string.Format("/configuration/appSettings/add[@key='{0}']", name)) as XmlElement;
                if (xmlElement != null)
                {
                    if (xmlElement.HasAttribute("value"))
                    {
                        text = xmlElement.GetAttribute("value");
                    }
                    if (flag && !string.IsNullOrEmpty(text))
                    {
                        text = Environment.ExpandEnvironmentVariables(text);
                    }
                    if (text != null)
                    {
                        string result = text;
                        return result;
                    }
                }
            }
            catch (Exception ex)
            {
                try
                {
                    Trace.WriteLine(string.Format(CultureInfo.CurrentCulture, "Native library pre-loader failed to get setting \"{0}\" value: {1}", new object[]
                    {
                        name,
                        ex
                    }));
                }
                catch
                {
                }
            }
            return @default;
        }

        private static string ListToString(IList<string> list)
        {
            if (list == null)
            {
                return null;
            }
            StringBuilder stringBuilder = new StringBuilder();
            foreach (string current in list)
            {
                if (current != null)
                {
                    if (stringBuilder.Length > 0)
                    {
                        stringBuilder.Append(' ');
                    }
                    stringBuilder.Append(current);
                }
            }
            return stringBuilder.ToString();
        }

        private static int CheckForArchitecturesAndPlatforms(string directory, ref List<string> matches)
        {
            int num = 0;
            if (matches == null)
            {
                matches = new List<string>();
            }
            lock (FFTWPlatformsAdapter.staticSyncRoot)
            {
                if (!string.IsNullOrEmpty(directory) && FFTWPlatformsAdapter.processorArchitecturePlatforms != null)
                {
                    foreach (KeyValuePair<string, string> current in FFTWPlatformsAdapter.processorArchitecturePlatforms)
                    {
                        if (Directory.Exists(Path.Combine(directory, current.Key)))
                        {
                            matches.Add(current.Key);
                            num++;
                        }
                        string value = current.Value;
                        if (value != null && Directory.Exists(Path.Combine(directory, value)))
                        {
                            matches.Add(value);
                            num++;
                        }
                    }
                }
            }
            return num;
        }

        private static bool CheckAssemblyCodeBase(Assembly assembly, ref string fileName)
        {
            try
            {
                bool result;
                if (assembly == null)
                {
                    result = false;
                    return result;
                }
                string codeBase = assembly.CodeBase;
                if (string.IsNullOrEmpty(codeBase))
                {
                    result = false;
                    return result;
                }
                Uri uri = new Uri(codeBase);
                string localPath = uri.LocalPath;
                if (!File.Exists(localPath))
                {
                    result = false;
                    return result;
                }
                string directoryName = Path.GetDirectoryName(localPath);
                string path = Path.Combine(directoryName, FFTWPlatformsAdapter.XmlConfigFileName);
                if (File.Exists(path))
                {
                    fileName = localPath;
                    result = true;
                    return result;
                }
                List<string> list = null;
                if (FFTWPlatformsAdapter.CheckForArchitecturesAndPlatforms(directoryName, ref list) > 0)
                {
                    fileName = localPath;
                    result = true;
                    return result;
                }
                result = false;
                return result;
            }
            catch (Exception ex)
            {
                try
                {
                    Trace.WriteLine(string.Format(CultureInfo.CurrentCulture, "Native library pre-loader failed to check code base for currently executing assembly: {0}", new object[]
                    {
                        ex
                    }));
                }
                catch
                {
                }
            }
            return false;
        }

        /// <summary>
        /// Queries and returns the directory for the assembly currently being
        /// executed.
        /// </summary>
        /// <returns>
        /// The directory for the assembly currently being executed -OR- null if
        /// it cannot be determined.
        /// </returns>
        private static string GetAssemblyDirectory()
        {
            try
            {
                Assembly executingAssembly = Assembly.GetExecutingAssembly();
                string result;
                if (executingAssembly == null)
                {
                    result = null;
                    return result;
                }
                string text = null;
                if (!FFTWPlatformsAdapter.CheckAssemblyCodeBase(executingAssembly, ref text))
                {
                    text = executingAssembly.Location;
                }
                if (string.IsNullOrEmpty(text))
                {
                    result = null;
                    return result;
                }
                string directoryName = Path.GetDirectoryName(text);
                if (string.IsNullOrEmpty(directoryName))
                {
                    result = null;
                    return result;
                }
                result = directoryName;
                return result;
            }
            catch (Exception ex)
            {
                try
                {
                    Trace.WriteLine(string.Format(CultureInfo.CurrentCulture, "Native library pre-loader failed to get directory for currently executing assembly: {0}", new object[]
                    {
                        ex
                    }));
                }
                catch
                {
                }
            }
            return null;
        }

        /// <summary>
        /// This is the P/Invoke method that wraps the native Win32 LoadLibrary
        /// function.  See the MSDN documentation for full details on what it
        /// does.
        /// </summary>
        /// <param name="fileName">
        /// The name of the executable library.
        /// </param>
        /// <returns>
        /// The native module handle upon success -OR- IntPtr.Zero on failure.
        /// </returns>
        [DllImport("kernel32", BestFitMapping = false, CharSet = CharSet.Auto, SetLastError = true, ThrowOnUnmappableChar = true)]
        private static extern IntPtr LoadLibrary(string fileName);

        /// <summary>
        /// Searches for the native fftw library in the directory containing
        /// the assembly currently being executed as well as the base directory
        /// for the current application domain.
        /// </summary>
        /// <param name="baseDirectory">
        /// Upon success, this parameter will be modified to refer to the base
        /// directory containing the native fftw library.
        /// </param>
        /// <param name="processorArchitecture">
        /// Upon success, this parameter will be modified to refer to the name
        /// of the immediate directory (i.e. the offset from the base directory)
        /// containing the native fftw library.
        /// </param>
        /// <returns>
        /// Non-zero (success) if the native fftw library was found; otherwise,
        /// zero (failure).
        /// </returns>
        private static bool SearchForDirectory(ref string baseDirectory, ref string processorArchitecture)
        {
            if (FFTWPlatformsAdapter.GetSettingValue("PreLoadfftw_NoSearchForDirectory", null) != null)
            {
                return false;
            }
            string[] array = new string[]
            {
                FFTWPlatformsAdapter.GetAssemblyDirectory(),
                AppDomain.CurrentDomain.BaseDirectory
            };
            string[] array2 = new string[]
            {
                FFTWPlatformsAdapter.GetProcessorArchitecture(),
                FFTWPlatformsAdapter.GetPlatformName(null)
            };
            string[] array3 = array;
            for (int i = 0; i < array3.Length; i++)
            {
                string text = array3[i];
                if (text != null)
                {
                    string[] array4 = array2;
                    for (int j = 0; j < array4.Length; j++)
                    {
                        string text2 = array4[j];
                        if (text2 != null)
                        {
                            string path = FFTWPlatformsAdapter.FixUpDllFileName(Path.Combine(Path.Combine(text, text2), "fftw.Interop.dll"));
                            if (File.Exists(path))
                            {
                                baseDirectory = text;
                                processorArchitecture = text2;
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Queries and returns the base directory of the current application
        /// domain.
        /// </summary>
        /// <returns>
        /// The base directory for the current application domain -OR- null if it
        /// cannot be determined.
        /// </returns>
        private static string GetBaseDirectory()
        {
            string text = FFTWPlatformsAdapter.GetSettingValue("PreLoadfftw_BaseDirectory", null);
            if (text != null)
            {
                return text;
            }
            if (FFTWPlatformsAdapter.GetSettingValue("PreLoadfftw_UseAssemblyDirectory", null) != null)
            {
                text = FFTWPlatformsAdapter.GetAssemblyDirectory();
                if (text != null)
                {
                    return text;
                }
            }
            return AppDomain.CurrentDomain.BaseDirectory;
        }

        /// <summary>
        /// Determines if the dynamic link library file name requires a suffix
        /// and adds it if necessary.
        /// </summary>
        /// <param name="fileName">
        /// The original dynamic link library file name to inspect.
        /// </param>
        /// <returns>
        /// The dynamic link library file name, possibly modified to include an
        /// extension.
        /// </returns>
        private static string FixUpDllFileName(string fileName)
        {
            if (!string.IsNullOrEmpty(fileName))
            {
                PlatformID platform = Environment.OSVersion.Platform;
                if ((platform == PlatformID.Win32S || platform == PlatformID.Win32Windows || platform == PlatformID.Win32NT || platform == PlatformID.WinCE) && !fileName.EndsWith(FFTWPlatformsAdapter.DllFileExtension, StringComparison.OrdinalIgnoreCase))
                {
                    return fileName + FFTWPlatformsAdapter.DllFileExtension;
                }
            }
            return fileName;
        }

        /// <summary>
        /// Queries and returns the processor architecture of the current
        /// process.
        /// </summary>
        /// <returns>
        /// The processor architecture of the current process -OR- null if it
        /// cannot be determined.
        /// </returns>
        private static string GetProcessorArchitecture()
        {
            string text = FFTWPlatformsAdapter.GetSettingValue("PreLoadfftw_ProcessorArchitecture", null);
            if (text != null)
            {
                return text;
            }
            text = FFTWPlatformsAdapter.GetSettingValue(FFTWPlatformsAdapter.PROCESSOR_ARCHITECTURE, null);
            if (IntPtr.Size == 4 && string.Equals(text, "AMD64", StringComparison.OrdinalIgnoreCase))
            {
                text = "x86";
            }
            return text;
        }

        /// <summary>
        /// Given the processor architecture, returns the name of the platform.
        /// </summary>
        /// <param name="processorArchitecture">
        /// The processor architecture to be translated to a platform name.
        /// </param>
        /// <returns>
        /// The platform name for the specified processor architecture -OR- null
        /// if it cannot be determined.
        /// </returns>
        private static string GetPlatformName(string processorArchitecture)
        {
            if (processorArchitecture == null)
            {
                processorArchitecture = FFTWPlatformsAdapter.GetProcessorArchitecture();
            }
            if (string.IsNullOrEmpty(processorArchitecture))
            {
                return null;
            }
            lock (FFTWPlatformsAdapter.staticSyncRoot)
            {
                if (FFTWPlatformsAdapter.processorArchitecturePlatforms == null)
                {
                    string result = null;
                    return result;
                }
                string text;
                if (FFTWPlatformsAdapter.processorArchitecturePlatforms.TryGetValue(processorArchitecture, out text))
                {
                    string result = text;
                    return result;
                }
            }
            return null;
        }

        /// <summary>
        /// Attempts to load the native fftw library based on the specified
        /// directory and processor architecture.
        /// </summary>
        /// <param name="baseDirectory">
        /// The base directory to use, null for default (the base directory of
        /// the current application domain).  This directory should contain the
        /// processor architecture specific sub-directories.
        /// </param>
        /// <param name="processorArchitecture">
        /// The requested processor architecture, null for default (the
        /// processor architecture of the current process).  This caller should
        /// almost always specify null for this parameter.
        /// </param>
        /// <param name="nativeModuleFileName">
        /// The candidate native module file name to load will be stored here,
        /// if necessary.
        /// </param>
        /// <param name="nativeModuleHandle">
        /// The native module handle as returned by LoadLibrary will be stored
        /// here, if necessary.  This value will be IntPtr.Zero if the call to
        /// LoadLibrary fails.
        /// </param>
        /// <returns>
        /// Non-zero if the native module was loaded successfully; otherwise,
        /// zero.
        /// </returns>
        private static bool PreLoadfftwDll(string baseDirectory, string processorArchitecture, ref string nativeModuleFileName, ref IntPtr nativeModuleHandle)
        {
            if (baseDirectory == null)
            {
                baseDirectory = FFTWPlatformsAdapter.GetBaseDirectory();
            }
            if (baseDirectory == null)
            {
                return false;
            }
            string text = FFTWPlatformsAdapter.FixUpDllFileName(Path.Combine(baseDirectory, fftw_DLL));
            if (File.Exists(text))
            {
                return false;
            }
            if (processorArchitecture == null)
            {
                processorArchitecture = FFTWPlatformsAdapter.GetProcessorArchitecture();
            }
            if (processorArchitecture == null)
            {
                return false;
            }
            text = FFTWPlatformsAdapter.FixUpDllFileName(Path.Combine(Path.Combine(baseDirectory, processorArchitecture), fftw_DLL));
            if (!File.Exists(text))
            {
                string platformName = FFTWPlatformsAdapter.GetPlatformName(processorArchitecture);
                if (platformName == null)
                {
                    return false;
                }
                text = FFTWPlatformsAdapter.FixUpDllFileName(Path.Combine(Path.Combine(baseDirectory, platformName), fftw_DLL));
                if (!File.Exists(text))
                {
                    return false;
                }
            }
            try
            {
                try
                {
                    Trace.WriteLine(string.Format(CultureInfo.CurrentCulture, "Native library pre-loader is trying to load native fftw library \"{0}\"...", new object[]
                    {
                        text
                    }));
                }
                catch
                {
                }
                nativeModuleFileName = text;
                nativeModuleHandle = FFTWPlatformsAdapter.LoadLibrary(text);
                return nativeModuleHandle != IntPtr.Zero;
            }
            catch (Exception ex)
            {
                try
                {
                    int lastWin32Error = Marshal.GetLastWin32Error();
                    Trace.WriteLine(string.Format(CultureInfo.CurrentCulture, "Native library pre-loader failed to load native fftw library \"{0}\" (getLastError = {1}): {2}", new object[]
                    {
                        text,
                        lastWin32Error,
                        ex
                    }));
                }
                catch
                {
                }
            }
            return false;
        }
    }
}
