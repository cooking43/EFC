using System;

/// <summary>
/// N2: 纯 C# 波浪谱计算类（无 MonoBehaviour 依赖）
///
/// 功能：
/// - JONSWAP/PM 谱函数 S(ω)
/// - 二阶 RAO 传递函数 H(ω)
/// - 随机相位叠加的分量振幅与相位生成
/// - 幅度标定（零阶矩 → 缩放系数）
/// - 给定 t 求 θ、θ̇、θ̈（含 smoothstep 爬升的乘积法则）
/// </summary>
public static class WaveSpectrum
{
    private const float AMPLITUDE_RAMP_DURATION = 0.5f;  // 平滑爬升时长

    /// <summary>
    /// 波浪谱模式
    /// </summary>
    public enum SpectrumMode
    {
        JONSWAP,
        PiersonMoskowitz,
        RegularSine
    }

    /// <summary>
    /// 波浪谱参数
    /// </summary>
    public struct WaveParameters
    {
        public SpectrumMode mode;

        // Irregular wave parameters
        public float significantAmplitude;  // Hs (degrees for roll/pitch, meters for heave)
        public float peakPeriod;            // Tp (seconds)
        public float gamma;                 // JONSWAP peak enhancement factor

        // RAO parameters
        public float naturalPeriod;         // Natural period (seconds)
        public float dampingRatio;          // Damping ratio ζ

        // Discretization
        public int numComponents;
        public float omegaMin;
        public float omegaMax;
        public int seed;

        // Regular wave mode
        public float regularAmplitude;
        public float regularFrequency;
        public float regularPhase;
    }

    /// <summary>
    /// 波浪状态（预计算的分量）
    /// </summary>
    public class WaveState
    {
        public float[] amplitudes;
        public float[] omegas;
        public float[] phases;
        public float episodeStartTime;

        public WaveState(int numComponents)
        {
            amplitudes = new float[numComponents];
            omegas = new float[numComponents];
            phases = new float[numComponents];
        }
    }

    /// <summary>
    /// JONSWAP 谱密度函数
    /// </summary>
    public static float JONSWAP(float omega, float Hs, float omegaPeak, float gamma)
    {
        if (omega <= 0) return 0f;

        float alpha = 5f / 16f * Hs * Hs * MathF.Pow(omegaPeak, 4f);
        float exp1 = MathF.Exp(-1.25f * MathF.Pow(omegaPeak / omega, 4f));
        float PMSpectrum = alpha / MathF.Pow(omega, 5f) * exp1;

        float sigma = (omega <= omegaPeak) ? 0.07f : 0.09f;
        float exp2 = MathF.Exp(-MathF.Pow(omega - omegaPeak, 2f) / (2f * sigma * sigma * omegaPeak * omegaPeak));
        float peakEnhancement = MathF.Pow(gamma, exp2);

        return PMSpectrum * peakEnhancement;
    }

    /// <summary>
    /// Pierson-Moskowitz 谱密度函数
    /// </summary>
    public static float PiersonMoskowitz(float omega, float Hs, float omegaPeak)
    {
        if (omega <= 0) return 0f;

        float alpha = 5f / 16f * Hs * Hs * MathF.Pow(omegaPeak, 4f);
        return alpha / MathF.Pow(omega, 5f) * MathF.Exp(-1.25f * MathF.Pow(omegaPeak / omega, 4f));
    }

    /// <summary>
    /// 二阶 RAO 传递函数幅值 |H(ω)|
    /// </summary>
    public static float RAOAmplitude(float omega, float omegaN, float zeta)
    {
        if (omega <= 0) return 0f;

        float omegaRatio = omega / omegaN;
        float omegaRatio2 = omegaRatio * omegaRatio;
        float denominator = MathF.Sqrt((1f - omegaRatio2) * (1f - omegaRatio2) + (2f * zeta * omegaRatio) * (2f * zeta * omegaRatio));

        return 1f / denominator;
    }

    /// <summary>
    /// 生成波浪状态（随机相位叠加）
    /// </summary>
    public static WaveState GenerateWaveState(WaveParameters waveParams, float currentTime)
    {
        WaveState state = new WaveState(waveParams.numComponents);
        state.episodeStartTime = currentTime;

        if (waveParams.mode == SpectrumMode.RegularSine)
        {
            // Regular wave: single component
            state.amplitudes = new float[1];
            state.omegas = new float[1];
            state.phases = new float[1];
            state.amplitudes[0] = waveParams.regularAmplitude;
            state.omegas[0] = waveParams.regularFrequency * 2f * MathF.PI;
            state.phases[0] = waveParams.regularPhase;
            return state;
        }

        // Irregular wave: JONSWAP or PM
        Random rng = (waveParams.seed < 0) ? new Random() : new Random(waveParams.seed);

        float omegaPeak = 2f * MathF.PI / waveParams.peakPeriod;
        float omegaN = 2f * MathF.PI / waveParams.naturalPeriod;

        float deltaOmega = (waveParams.omegaMax - waveParams.omegaMin) / waveParams.numComponents;

        float sumS = 0f;
        for (int i = 0; i < waveParams.numComponents; i++)
        {
            float omega = waveParams.omegaMin + (i + 0.5f) * deltaOmega;
            state.omegas[i] = omega;

            // Spectrum density
            float S_omega = (waveParams.mode == SpectrumMode.JONSWAP)
                ? JONSWAP(omega, waveParams.significantAmplitude, omegaPeak, waveParams.gamma)
                : PiersonMoskowitz(omega, waveParams.significantAmplitude, omegaPeak);

            // RAO transfer function
            float H_omega = RAOAmplitude(omega, omegaN, waveParams.dampingRatio);

            // Component amplitude (before calibration)
            float amplitude = MathF.Sqrt(2f * S_omega * H_omega * H_omega * deltaOmega);
            state.amplitudes[i] = amplitude;

            // Random phase
            state.phases[i] = (float)(rng.NextDouble() * 2.0 * Math.PI);

            // N5: Accumulate variance including RAO (must match amplitude formula)
            sumS += S_omega * H_omega * H_omega * deltaOmega;
        }

        // Calibrate to match significant amplitude (m0 = Hs²/16 for amplitude spectrum)
        float m0_target = waveParams.significantAmplitude * waveParams.significantAmplitude / 16f;
        float m0_actual = sumS;
        float scaleFactor = (m0_actual > 1e-6f) ? MathF.Sqrt(m0_target / m0_actual) : 1f;

        for (int i = 0; i < waveParams.numComponents; i++)
        {
            state.amplitudes[i] *= scaleFactor;
        }

        return state;
    }

    /// <summary>
    /// 计算给定时刻的角度、角速度、角加速度（含 smootherstep 爬升，C² 连续）
    /// </summary>
    public static void EvaluateMotion(WaveState state, float currentTime,
        out float angle, out float angularVelocity, out float angularAcceleration)
    {
        float t = currentTime - state.episodeStartTime;

        // O3b: Smootherstep ramp (C² continuous): r(t) = 6t⁵ - 15t⁴ + 10t³
        // r'(t) = 30t⁴ - 60t³ + 30t², r''(t) = 120t³ - 180t² + 60t
        // r'(0) = r'(1) = 0, r''(0) = r''(1) = 0 (no discontinuity)
        float rampFactor = 1f;
        float rampDerivative = 0f;
        float rampSecondDerivative = 0f;

        if (t < AMPLITUDE_RAMP_DURATION)
        {
            float t01 = t / AMPLITUDE_RAMP_DURATION;
            float t2 = t01 * t01;
            float t3 = t2 * t01;
            float t4 = t3 * t01;
            float t5 = t4 * t01;

            rampFactor = 6f * t5 - 15f * t4 + 10f * t3;
            rampDerivative = (30f * t4 - 60f * t3 + 30f * t2) / AMPLITUDE_RAMP_DURATION;
            rampSecondDerivative = (120f * t3 - 180f * t2 + 60f * t01) / (AMPLITUDE_RAMP_DURATION * AMPLITUDE_RAMP_DURATION);
        }

        // Base wave (without ramp)
        float angleBase = 0f;
        float velocityBase = 0f;
        float accelerationBase = 0f;

        for (int i = 0; i < state.amplitudes.Length; i++)
        {
            float omega = state.omegas[i];
            float phase = omega * t + state.phases[i];
            float A = state.amplitudes[i];

            angleBase += A * MathF.Cos(phase);
            velocityBase += -A * omega * MathF.Sin(phase);
            accelerationBase += -A * omega * omega * MathF.Cos(phase);
        }

        // Product rule: θ = r·s, θ̇ = r'·s + r·ṡ, θ̈ = r''·s + 2r'·ṡ + r·s̈
        angle = rampFactor * angleBase;
        angularVelocity = rampDerivative * angleBase + rampFactor * velocityBase;
        angularAcceleration = rampSecondDerivative * angleBase + 2f * rampDerivative * velocityBase + rampFactor * accelerationBase;
    }

    /// <summary>
    /// 计算谱的零阶矩 m0（用于幅度标定验证）
    /// </summary>
    public static float ComputeZerothMoment(WaveParameters waveParams)
    {
        float omegaPeak = 2f * MathF.PI / waveParams.peakPeriod;
        float deltaOmega = (waveParams.omegaMax - waveParams.omegaMin) / waveParams.numComponents;

        float m0 = 0f;
        for (int i = 0; i < waveParams.numComponents; i++)
        {
            float omega = waveParams.omegaMin + (i + 0.5f) * deltaOmega;
            float S_omega = (waveParams.mode == SpectrumMode.JONSWAP)
                ? JONSWAP(omega, waveParams.significantAmplitude, omegaPeak, waveParams.gamma)
                : PiersonMoskowitz(omega, waveParams.significantAmplitude, omegaPeak);
            m0 += S_omega * deltaOmega;
        }

        return m0;
    }
}
