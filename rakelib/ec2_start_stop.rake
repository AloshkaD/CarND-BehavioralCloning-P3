INSTANCE_ID = 'i-516ff7c4'

namespace :carnd do
  task :start, [:instance_id] do |t, args|
    args.with_defaults(instance_id: INSTANCE_ID)
    instance_id = args[:instance_id]
    sh "aws ec2 start-instances --instance-ids \"#{instance_id}\""
  end

  task :stop, [:instance_id] do |t, args|
    args.with_defaults(instance_id: INSTANCE_ID)
    instance_id = args[:instance_id]
    sh "aws ec2 stop-instances --instance-ids \"#{instance_id}\""
  end
end