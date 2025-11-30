//! Pacha CLI - Model, Data and Recipe Registry

use clap::{Parser, Subcommand};
use pacha::prelude::*;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "pacha")]
#[command(author, version, about = "Model, Data and Recipe Registry", long_about = None)]
struct Cli {
    /// Registry path (default: ~/.pacha)
    #[arg(long, global = true)]
    registry: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Model registry operations
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
    /// Dataset registry operations
    Data {
        #[command(subcommand)]
        action: DataAction,
    },
    /// Recipe registry operations
    Recipe {
        #[command(subcommand)]
        action: RecipeAction,
    },
    /// Experiment run operations
    Run {
        #[command(subcommand)]
        action: RunAction,
    },
    /// Show registry statistics
    Stats,
    /// Initialize a new registry
    Init,
}

#[derive(Subcommand)]
enum ModelAction {
    /// Register a new model
    Register {
        /// Model name
        name: String,
        /// Path to artifact file
        artifact: PathBuf,
        /// Model version (e.g., 1.0.0)
        #[arg(long, short)]
        version: String,
        /// Model description
        #[arg(long, short)]
        description: Option<String>,
    },
    /// List models
    List {
        /// Model name (optional, lists versions if provided)
        name: Option<String>,
    },
    /// Get model details
    Get {
        /// Model name
        name: String,
        /// Model version
        #[arg(long, short)]
        version: String,
    },
    /// Download model artifact
    Download {
        /// Model name
        name: String,
        /// Model version
        #[arg(long, short)]
        version: String,
        /// Output path
        #[arg(long, short)]
        output: PathBuf,
    },
    /// Show model lineage
    Lineage {
        /// Model name
        name: String,
        /// Model version
        #[arg(long, short)]
        version: String,
    },
    /// Transition model stage
    Stage {
        /// Model name
        name: String,
        /// Model version
        #[arg(long, short)]
        version: String,
        /// Target stage (development, staging, production, archived)
        #[arg(long, short)]
        target: String,
    },
}

#[derive(Subcommand)]
enum DataAction {
    /// Register a new dataset
    Register {
        /// Dataset name
        name: String,
        /// Path to data file
        data: PathBuf,
        /// Dataset version (e.g., 1.0.0)
        #[arg(long, short)]
        version: String,
        /// Dataset purpose/description
        #[arg(long, short)]
        purpose: Option<String>,
    },
    /// List datasets
    List {
        /// Dataset name (optional)
        name: Option<String>,
    },
    /// Get dataset details
    Get {
        /// Dataset name
        name: String,
        /// Dataset version
        #[arg(long, short)]
        version: String,
    },
    /// Download dataset
    Download {
        /// Dataset name
        name: String,
        /// Dataset version
        #[arg(long, short)]
        version: String,
        /// Output path
        #[arg(long, short)]
        output: PathBuf,
    },
}

#[derive(Subcommand)]
enum RecipeAction {
    /// Register a new recipe
    Register {
        /// Path to recipe TOML file
        recipe: PathBuf,
    },
    /// List recipes
    List {
        /// Recipe name (optional)
        name: Option<String>,
    },
    /// Get recipe details
    Get {
        /// Recipe name
        name: String,
        /// Recipe version
        #[arg(long, short)]
        version: String,
    },
    /// Validate a recipe
    Validate {
        /// Recipe name
        name: String,
        /// Recipe version
        #[arg(long, short)]
        version: String,
    },
}

#[derive(Subcommand)]
enum RunAction {
    /// List experiment runs
    List {
        /// Recipe name
        recipe: String,
        /// Recipe version
        #[arg(long, short)]
        version: String,
    },
    /// Get run details
    Get {
        /// Run ID
        id: String,
    },
    /// Compare runs
    Compare {
        /// Run IDs to compare
        ids: Vec<String>,
    },
    /// Find best run by metric
    Best {
        /// Recipe name
        recipe: String,
        /// Recipe version
        #[arg(long, short)]
        version: String,
        /// Metric name
        #[arg(long, short)]
        metric: String,
        /// Minimize metric (default: maximize)
        #[arg(long)]
        minimize: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("Error: {e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn run(cli: Cli) -> pacha::Result<()> {
    let config = cli.registry.map(RegistryConfig::new).unwrap_or_default();

    match cli.command {
        Commands::Init => {
            let registry = Registry::open(config)?;
            println!(
                "Registry initialized at: {}",
                registry.config().base_path.display()
            );
        }
        Commands::Stats => {
            let registry = Registry::open(config)?;
            let stats = registry.storage_stats()?;
            println!("Registry Statistics:");
            println!("  Models:   {}", stats.model_count);
            println!("  Datasets: {}", stats.dataset_count);
            println!("  Recipes:  {}", stats.recipe_count);
            println!("  Objects:  {}", stats.object_count);
            println!("  Size:     {} bytes", stats.total_size_bytes);
        }
        Commands::Model { action } => handle_model(config, action)?,
        Commands::Data { action } => handle_data(config, action)?,
        Commands::Recipe { action } => handle_recipe(config, action)?,
        Commands::Run { action } => handle_run(config, action)?,
    }

    Ok(())
}

fn handle_model(config: RegistryConfig, action: ModelAction) -> pacha::Result<()> {
    let registry = Registry::open(config)?;

    match action {
        ModelAction::Register {
            name,
            artifact,
            version,
            description,
        } => {
            let version: ModelVersion = version.parse()?;
            let data = std::fs::read(&artifact)?;
            let card = ModelCard::new(description.unwrap_or_default());
            let id = registry.register_model(&name, &version, &data, card)?;
            println!("Registered model: {name}:{version} ({id})");
        }
        ModelAction::List { name } => {
            if let Some(name) = name {
                let versions = registry.list_model_versions(&name)?;
                println!("Versions of '{name}':");
                for v in versions {
                    println!("  {v}");
                }
            } else {
                let models = registry.list_models()?;
                println!("Models:");
                for m in models {
                    println!("  {m}");
                }
            }
        }
        ModelAction::Get { name, version } => {
            let version: ModelVersion = version.parse()?;
            let model = registry.get_model(&name, &version)?;
            println!("Model: {}:{}", model.name, model.version);
            println!("  ID:          {}", model.id);
            println!("  Stage:       {}", model.stage);
            println!("  Created:     {}", model.created_at);
            println!("  Description: {}", model.card.description);
            println!("  Size:        {} bytes", model.content_address.size());
            println!("  Hash:        {}", model.content_address.hash_hex());
            if !model.card.metrics.is_empty() {
                println!("  Metrics:");
                for (k, v) in &model.card.metrics {
                    println!("    {k}: {v}");
                }
            }
        }
        ModelAction::Download {
            name,
            version,
            output,
        } => {
            let version: ModelVersion = version.parse()?;
            let data = registry.get_model_artifact(&name, &version)?;
            std::fs::write(&output, &data)?;
            println!("Downloaded {name}:{version} to {}", output.display());
        }
        ModelAction::Lineage { name, version } => {
            let version: ModelVersion = version.parse()?;
            let model = registry.get_model(&name, &version)?;
            let lineage = registry.get_model_lineage(&model.id)?;
            println!("Lineage for {name}:{version}:");
            println!("  Nodes: {}", lineage.node_count());
            println!("  Edges: {}", lineage.edge_count());
            // TODO: Better lineage visualization
        }
        ModelAction::Stage {
            name,
            version,
            target,
        } => {
            let version: ModelVersion = version.parse()?;
            let target_stage: ModelStage = target.parse()?;
            registry.transition_model_stage(&name, &version, target_stage)?;
            println!("Transitioned {name}:{version} to {target_stage}");
        }
    }

    Ok(())
}

fn handle_data(config: RegistryConfig, action: DataAction) -> pacha::Result<()> {
    let registry = Registry::open(config)?;

    match action {
        DataAction::Register {
            name,
            data,
            version,
            purpose,
        } => {
            let version: DatasetVersion = version.parse()?;
            let content = std::fs::read(&data)?;
            let datasheet = Datasheet::new(purpose.unwrap_or_default());
            let id = registry.register_dataset(&name, &version, &content, datasheet)?;
            println!("Registered dataset: {name}:{version} ({id})");
        }
        DataAction::List { name } => {
            if name.is_some() {
                // TODO: list versions for a specific dataset
                println!("Version listing not yet implemented");
            } else {
                let datasets = registry.list_datasets()?;
                println!("Datasets:");
                for d in datasets {
                    println!("  {d}");
                }
            }
        }
        DataAction::Get { name, version } => {
            let version: DatasetVersion = version.parse()?;
            let dataset = registry.get_dataset(&name, &version)?;
            println!("Dataset: {}:{}", dataset.name, dataset.version);
            println!("  ID:      {}", dataset.id);
            println!("  Created: {}", dataset.created_at);
            println!("  Purpose: {}", dataset.datasheet.purpose);
            println!("  Size:    {} bytes", dataset.content_address.size());
            println!("  Hash:    {}", dataset.content_address.hash_hex());
        }
        DataAction::Download {
            name,
            version,
            output,
        } => {
            let version: DatasetVersion = version.parse()?;
            let data = registry.get_dataset_data(&name, &version)?;
            std::fs::write(&output, &data)?;
            println!("Downloaded {name}:{version} to {}", output.display());
        }
    }

    Ok(())
}

fn handle_recipe(config: RegistryConfig, action: RecipeAction) -> pacha::Result<()> {
    let registry = Registry::open(config)?;

    match action {
        RecipeAction::Register { recipe: path } => {
            let content = std::fs::read_to_string(&path)?;
            let recipe: TrainingRecipe = toml::from_str(&content)?;
            let id = registry.register_recipe(&recipe)?;
            println!(
                "Registered recipe: {}:{} ({})",
                recipe.name, recipe.version, id
            );
        }
        RecipeAction::List { name } => {
            if name.is_some() {
                // TODO: list versions
                println!("Version listing not yet implemented");
            } else {
                let recipes = registry.list_recipes()?;
                println!("Recipes:");
                for r in recipes {
                    println!("  {r}");
                }
            }
        }
        RecipeAction::Get { name, version } => {
            let version: RecipeVersion = version.parse()?;
            let recipe = registry.get_recipe(&name, &version)?;
            println!("Recipe: {}:{}", recipe.name, recipe.version);
            println!("  ID:          {}", recipe.id);
            println!("  Description: {}", recipe.description);
            println!("  Created:     {}", recipe.created_at);
            println!("  Hyperparameters:");
            println!(
                "    Learning rate: {}",
                recipe.hyperparameters.learning_rate
            );
            println!("    Batch size:    {}", recipe.hyperparameters.batch_size);
            println!("    Epochs:        {}", recipe.hyperparameters.epochs);
        }
        RecipeAction::Validate { name, version } => {
            let version: RecipeVersion = version.parse()?;
            let recipe = registry.get_recipe(&name, &version)?;
            println!("Validating recipe: {name}:{version}...");
            // Basic validation
            if recipe.hyperparameters.learning_rate <= 0.0 {
                println!("  Warning: learning_rate should be positive");
            }
            if recipe.hyperparameters.batch_size == 0 {
                println!("  Warning: batch_size should be > 0");
            }
            println!("  Validation complete");
        }
    }

    Ok(())
}

fn handle_run(config: RegistryConfig, action: RunAction) -> pacha::Result<()> {
    let registry = Registry::open(config)?;

    match action {
        RunAction::List { recipe, version } => {
            let version: RecipeVersion = version.parse()?;
            let recipe_ref = RecipeReference::new(recipe, version);
            let runs = registry.list_runs(&recipe_ref)?;
            println!("Runs for {recipe_ref}:");
            for run in runs {
                println!(
                    "  {} - {} ({})",
                    run.run_id,
                    run.status,
                    run.started_at.format("%Y-%m-%d %H:%M:%S")
                );
            }
        }
        RunAction::Get { id } => {
            let run_id: RunId = id
                .parse()
                .map_err(|_| pacha::PachaError::Validation("invalid run id".to_string()))?;
            let run = registry.get_run(&run_id)?;
            println!("Run: {}", run.run_id);
            println!("  Status:  {}", run.status);
            println!("  Started: {}", run.started_at);
            if let Some(finished) = run.finished_at {
                println!("  Finished: {finished}");
            }
            if !run.metrics.is_empty() {
                println!("  Final metrics:");
                // Get latest value for each metric
                let mut latest: std::collections::HashMap<&str, f64> =
                    std::collections::HashMap::new();
                for m in &run.metrics {
                    latest.insert(&m.name, m.value);
                }
                for (k, v) in latest {
                    println!("    {k}: {v}");
                }
            }
        }
        RunAction::Compare { ids } => {
            println!("Comparing runs:");
            for id in ids {
                let run_id: RunId = id
                    .parse()
                    .map_err(|_| pacha::PachaError::Validation("invalid run id".to_string()))?;
                let run = registry.get_run(&run_id)?;
                println!("  {} - {}", run.run_id, run.status);
            }
            // TODO: Implement proper comparison
        }
        RunAction::Best {
            recipe,
            version,
            metric,
            minimize,
        } => {
            let version: RecipeVersion = version.parse()?;
            let recipe_ref = RecipeReference::new(recipe, version);
            let runs = registry.list_runs(&recipe_ref)?;

            let best = runs
                .iter()
                .filter(|r| r.status == RunStatus::Completed)
                .filter_map(|r| r.get_metric(&metric).map(|v| (r, v)))
                .min_by(|(_, a), (_, b)| {
                    if minimize {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    }
                });

            if let Some((run, value)) = best {
                println!("Best run by {metric}:");
                println!("  ID:    {}", run.run_id);
                println!("  Value: {value}");
            } else {
                println!("No completed runs found with metric '{metric}'");
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse() {
        let cli = Cli::try_parse_from(["pacha", "stats"]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_model_register() {
        let cli = Cli::try_parse_from([
            "pacha",
            "model",
            "register",
            "test-model",
            "model.apr",
            "-v",
            "1.0.0",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_model_list() {
        let cli = Cli::try_parse_from(["pacha", "model", "list"]);
        assert!(cli.is_ok());
    }
}
