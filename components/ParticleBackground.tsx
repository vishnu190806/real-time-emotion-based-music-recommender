import React, { useEffect, useRef } from 'react';

interface Particle {
    x: number;
    y: number;
    size: number;
    baseX: number;
    baseY: number;
    vx: number;
    vy: number;
    color: string;
}

const ParticleBackground: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let animationFrameId: number;
        let particles: Particle[] = [];
        
        // Settings
        const numParticles = Math.min(window.innerWidth / 2, 800); // Scale by screen size
        const colors = ['#60a5fa', '#818cf8', '#a78bfa', '#c084fc', '#e879f9']; // Blues to Purples (Techy/Glow)
        const interactionRadius = 150;
        const interactionForce = 0.05;

        // Mouse tracking
        let mouse = { x: -1000, y: -1000 };

        const handleMouseMove = (e: MouseEvent) => {
            mouse.x = e.clientX;
            mouse.y = e.clientY;
        };
        
        const handleMouseLeave = () => {
            mouse.x = -1000;
            mouse.y = -1000;
        };

        window.addEventListener('mousemove', handleMouseMove);
        document.body.addEventListener('mouseleave', handleMouseLeave);

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            initParticles();
        };

        window.addEventListener('resize', resize);

        const initParticles = () => {
            particles = [];
            for (let i = 0; i < numParticles; i++) {
                const x = Math.random() * canvas.width;
                const y = Math.random() * canvas.height;
                particles.push({
                    x,
                    y,
                    size: Math.random() * 2 + 0.5,
                    baseX: x,
                    baseY: y,
                    vx: 0, // Current velocity X
                    vy: 0, // Current velocity Y
                    color: colors[Math.floor(Math.random() * colors.length)]
                });
            }
        };

        const render = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height); // Keep it completely transparent to let app background show through

            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];

                // Distance to mouse
                const dx = mouse.x - p.x;
                const dy = mouse.y - p.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                // Interaction Logics: Repel
                if (distance < interactionRadius) {
                    const forceDirectionX = dx / distance;
                    const forceDirectionY = dy / distance;
                    
                    // The closer to the mouse, the stronger the force
                    const force = (interactionRadius - distance) / interactionRadius;
                    
                    p.vx -= forceDirectionX * force * 1.5;
                    p.vy -= forceDirectionY * force * 1.5;
                }

                // Spring back to base position
                p.vx += (p.baseX - p.x) * interactionForce;
                p.vy += (p.baseY - p.y) * interactionForce;

                // Friction
                p.vx *= 0.85;
                p.vy *= 0.85;

                p.x += p.vx;
                p.y += p.vy;

                // Glow effect occasionally
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = p.color;
                
                // Add soft glow
                ctx.shadowBlur = 10;
                ctx.shadowColor = p.color;
                
                ctx.fill();
                
                // Reset shadow for performance
                ctx.shadowBlur = 0; 
            }

            animationFrameId = requestAnimationFrame(render);
        };

        // Initialize
        resize();
        render();

        return () => {
            window.removeEventListener('resize', resize);
            window.removeEventListener('mousemove', handleMouseMove);
            document.body.removeEventListener('mouseleave', handleMouseLeave);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 pointer-events-none z-0"
            style={{ opacity: 0.6 }} // Adjust overall opacity so it doesn't overpower the UI
        />
    );
};

export default ParticleBackground;
