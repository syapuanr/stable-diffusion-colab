import gradio as gr
import torch
import random
from pathlib import Path
from ..core import ModelLoader, InferenceEngine, MemoryManager
from .components import (
    create_generation_settings,
    create_prompt_inputs,
    create_model_selector,
    create_lora_settings,
    create_advanced_settings,
    create_memory_monitor
)

class GradioInterface:
    """Main Gradio interface for Stable Diffusion WebUI"""
    
    def __init__(self, config):
        self.config = config
        self.model_loader = ModelLoader(config)
        self.inference_engine = InferenceEngine(self.model_loader)
        self.memory_manager = MemoryManager(config)
        
        # Output directory
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_model_handler(self, model_source, model_id, model_file, vae_file,
                          lora1, lora1_w, lora2, lora2_w, lora3, lora3_w):
        """Handle model loading"""
        try:
            # Prepare LoRA paths and weights
            lora_paths = []
            lora_weights = []
            
            for lora_file, weight in [(lora1, lora1_w), (lora2, lora2_w), (lora3, lora3_w)]:
                if lora_file is not None:
                    lora_paths.append(lora_file.name if hasattr(lora_file, 'name') else lora_file)
                    lora_weights.append(weight)
            
            # Load model
            if model_source == "HuggingFace":
                self.model_loader.load_model(
                    model_id=model_id,
                    vae_path=vae_file.name if vae_file else None,
                    lora_paths=lora_paths if lora_paths else None,
                    lora_weights=lora_weights if lora_weights else None
                )
                status = f"✓ Model loaded: {model_id}"
            else:
                if model_file is None:
                    return "✗ Please select a model file"
                
                self.model_loader.load_model(
                    model_path=model_file.name,
                    vae_path=vae_file.name if vae_file else None,
                    lora_paths=lora_paths if lora_paths else None,
                    lora_weights=lora_weights if lora_weights else None
                )
                status = f"✓ Model loaded: {Path(model_file.name).name}"
            
            # Get model info
            info = self.model_loader.get_current_model_info()
            status += f"\nType: {info['type'].upper()}"
            status += f"\nDevice: {info['device']}"
            status += f"\nDtype: {info['dtype']}"
            
            return status
            
        except Exception as e:
            return f"✗ Error loading model: {str(e)}"
    
    def generate_txt2img_handler(self, prompt, negative_prompt, width, height,
                                 steps, cfg_scale, num_images, seed, scheduler):
        """Handle text-to-image generation"""
        try:
            if not self.model_loader.current_pipeline:
                return None, "✗ Please load a model first"
            
            # Handle seed
            if seed == -1:
                seed = random.randint(0, 2147483647)
            
            # Set scheduler if changed
            current_scheduler = self.model_loader.current_pipeline.scheduler.__class__.__name__
            if scheduler not in current_scheduler:
                self.model_loader.set_scheduler(
                    self.model_loader.current_pipeline, 
                    scheduler
                )
            
            # Generate images
            images = self.inference_engine.generate_txt2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(width),
                height=int(height),
                num_inference_steps=int(steps),
                guidance_scale=cfg_scale,
                num_images=int(num_images),
                seed=int(seed)
            )
            
            # Save images
            saved_paths = self.inference_engine.save_images(
                images, 
                output_dir=str(self.output_dir),
                prefix="txt2img"
            )
            
            info = f"✓ Generated {len(images)} image(s)\n"
            info += f"Seed: {seed}\n"
            info += f"Saved to: {self.output_dir}"
            
            return images, info
            
        except Exception as e:
            import traceback
            error_msg = f"✗ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
    def generate_img2img_handler(self, init_image, prompt, negative_prompt,
                                strength, steps, cfg_scale, num_images, seed, scheduler):
        """Handle image-to-image generation"""
        try:
            if not self.model_loader.current_pipeline:
                return None, "✗ Please load a model first"
            
            if init_image is None:
                return None, "✗ Please provide an input image"
            
            # Handle seed
            if seed == -1:
                seed = random.randint(0, 2147483647)
            
            # Generate images
            images = self.inference_engine.generate_img2img(
                prompt=prompt,
                image=init_image,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=int(steps),
                guidance_scale=cfg_scale,
                num_images=int(num_images),
                seed=int(seed)
            )
            
            # Save images
            saved_paths = self.inference_engine.save_images(
                images,
                output_dir=str(self.output_dir),
                prefix="img2img"
            )
            
            info = f"✓ Generated {len(images)} image(s)\n"
            info += f"Seed: {seed}\n"
            info += f"Strength: {strength}\n"
            info += f"Saved to: {self.output_dir}"
            
            return images, info
            
        except Exception as e:
            import traceback
            error_msg = f"✗ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg
    
    def get_memory_stats_handler(self):
        """Get current memory statistics"""
        stats = []
        
        # GPU Memory
        gpu_info = self.memory_manager.get_gpu_memory_info()
        if gpu_info:
            stats.append("=== GPU Memory ===")
            stats.append(f"Allocated: {gpu_info['allocated']:.2f} GB")
            stats.append(f"Reserved:  {gpu_info['reserved']:.2f} GB")
            stats.append(f"Free:      {gpu_info['free']:.2f} GB")
            stats.append(f"Total:     {gpu_info['total']:.2f} GB")
            stats.append("")
        
        # RAM
        ram_info = self.memory_manager.get_ram_info()
        stats.append("=== System RAM ===")
        stats.append(f"Used:      {ram_info['used']:.2f} GB")
        stats.append(f"Available: {ram_info['available']:.2f} GB")
        stats.append(f"Total:     {ram_info['total']:.2f} GB")
        stats.append(f"Usage:     {ram_info['percent']:.1f}%")
        stats.append("")
        
        # Model info
        if self.model_loader.current_pipeline:
            info = self.model_loader.get_current_model_info()
            stats.append("=== Current Model ===")
            stats.append(f"Model: {info['model']}")
            stats.append(f"Type:  {info['type'].upper()}")
            stats.append(f"Device: {info['device']}")
        else:
            stats.append("=== Current Model ===")
            stats.append("No model loaded")
        
        return "\n".join(stats)
    
    def update_model_input_visibility(self, source):
        """Update visibility of model input fields based on source"""
        if source == "HuggingFace":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
    
    def create_interface(self):
        """Create the complete Gradio interface"""
        
        with gr.Blocks(
            title="Stable Diffusion WebUI",
            theme=gr.themes.Soft(),
            css="""
                .gradio-container {
                    max-width: 1200px !important;
                }
            """
        ) as demo:
            
            gr.Markdown(
                """
                # 🎨 Stable Diffusion WebUI
                ### Optimized for Google Colab
                """
            )
            
            # Model Selection and Settings
            (model_source, model_id, model_file, vae_file, 
             load_model_btn, model_status) = create_model_selector()
            
            # LoRA Settings
            (lora1_file, lora1_weight, lora2_file, lora2_weight,
             lora3_file, lora3_weight) = create_lora_settings()
            
            # Advanced Settings
            (enable_cpu_offload, enable_xformers, enable_vae_tiling,
             enable_attention_slicing, clip_skip) = create_advanced_settings()
            
            # Memory Monitor
            memory_stats, refresh_memory_btn = create_memory_monitor()
            
            # Main Tabs
            with gr.Tabs():
                
                # Text-to-Image Tab
                with gr.Tab("Text-to-Image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            prompt_t2i, negative_prompt_t2i = create_prompt_inputs()
                            
                            (width_t2i, height_t2i, steps_t2i, cfg_t2i,
                             num_images_t2i, seed_t2i, scheduler_t2i) = create_generation_settings()
                            
                            generate_t2i_btn = gr.Button(
                                "Generate",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            output_gallery_t2i = gr.Gallery(
                                label="Generated Images",
                                show_label=True,
                                elem_id="gallery",
                                columns=2,
                                rows=2,
                                object_fit="contain",
                                height="auto"
                            )
                            
                            output_info_t2i = gr.Textbox(
                                label="Generation Info",
                                lines=5,
                                interactive=False
                            )
                
                # Image-to-Image Tab
                with gr.Tab("Image-to-Image"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            init_image = gr.Image(
                                label="Input Image",
                                type="pil",
                                tool="editor"
                            )
                            
                            prompt_i2i, negative_prompt_i2i = create_prompt_inputs()
                            
                            strength = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.75,
                                label="Denoising Strength"
                            )
                            
                            (width_i2i, height_i2i, steps_i2i, cfg_i2i,
                             num_images_i2i, seed_i2i, scheduler_i2i) = create_generation_settings()
                            
                            generate_i2i_btn = gr.Button(
                                "Generate",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            output_gallery_i2i = gr.Gallery(
                                label="Generated Images",
                                show_label=True,
                                columns=2,
                                rows=2,
                                object_fit="contain",
                                height="auto"
                            )
                            
                            output_info_i2i = gr.Textbox(
                                label="Generation Info",
                                lines=5,
                                interactive=False
                            )
                
                # Settings Tab
                with gr.Tab("Settings"):
                    gr.Markdown("### Configuration")
                    gr.Markdown("Adjust your settings here. Changes will be applied on next model load.")
                    
                    with gr.Row():
                        save_config_btn = gr.Button("Save Configuration")
                        load_config_btn = gr.Button("Reload Configuration")
                    
                    config_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
            
            # Event Handlers
            
            # Model source change
            model_source.change(
                fn=self.update_model_input_visibility,
                inputs=[model_source],
                outputs=[model_id, model_file]
            )
            
            # Load model
            load_model_btn.click(
                fn=self.load_model_handler,
                inputs=[
                    model_source, model_id, model_file, vae_file,
                    lora1_file, lora1_weight,
                    lora2_file, lora2_weight,
                    lora3_file, lora3_weight
                ],
                outputs=[model_status]
            )
            
            # Text-to-Image generation
            generate_t2i_btn.click(
                fn=self.generate_txt2img_handler,
                inputs=[
                    prompt_t2i, negative_prompt_t2i,
                    width_t2i, height_t2i, steps_t2i, cfg_t2i,
                    num_images_t2i, seed_t2i, scheduler_t2i
                ],
                outputs=[output_gallery_t2i, output_info_t2i]
            )
            
            # Image-to-Image generation
            generate_i2i_btn.click(
                fn=self.generate_img2img_handler,
                inputs=[
                    init_image, prompt_i2i, negative_prompt_i2i,
                    strength, steps_i2i, cfg_i2i,
                    num_images_i2i, seed_i2i, scheduler_i2i
                ],
                outputs=[output_gallery_i2i, output_info_i2i]
            )
            
            # Memory stats refresh
            refresh_memory_btn.click(
                fn=self.get_memory_stats_handler,
                outputs=[memory_stats]
            )
            
            # Load memory stats on startup
            demo.load(
                fn=self.get_memory_stats_handler,
                outputs=[memory_stats]
            )
        
        return demo

def create_interface(config):
    """Factory function to create Gradio interface"""
    interface = GradioInterface(config)
    return interface.create_interface()